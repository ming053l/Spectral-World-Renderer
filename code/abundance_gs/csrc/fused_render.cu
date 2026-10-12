// Fused per-hit kernels of the fast renderer (abundance_gs/model.py: render_view_fused).
//
//   decode     per kept hit, fully fused: tetrahedral interpolation of the 4x12 primitive features, sin/cos encoding
//              (24) and the degree-2 SH code of the view direction (9), then the 4-layer MLP in fp16 on tensor cores
//              (WMMA, fp32 accumulation, bias + SiLU in fp32) -> raw outputs y [M, K+1] in fp16. 128 hits per block;
//              activations live in shared memory, the weights are read through L1.
//   composite  per pixel (one warp, hits contiguous per pixel): softmax of the K abundance logits, softplus intensity,
//              blend-weighted accumulation in hit order -> abundance [P,K] (normalized by alpha), sum w*s*a [P,K], alpha [P].
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kHits = 128;   // hits per decode block (8 warps x 16 rows)
constexpr int kLd = 136;     // shared-memory row stride (halves) of activations and weights
constexpr int kIn = 48;      // padded input width of the first layer (3 WMMA k-tiles)

__device__ __forceinline__ void encode_row(const float* __restrict__ features, const float* __restrict__ means,
                                           long id, float x, float y, float z, float cx, float cy, float cz, __half* row) {
  const float s6 = 0.20412414523193150f, s2 = 0.11785113019775792f, s2b = 0.23570226039551584f;
  const float w[4] = {.25f + s6 * x - s2 * y - z / 12.f, .25f - s6 * x - s2 * y - z / 12.f,
                      .25f + s2b * y - z / 12.f, .25f + z / 4.f};
  const float* f = features + id * 48;
  #pragma unroll
  for (int c = 0; c < 12; ++c) {
    float v = f[c] * w[0] + f[12 + c] * w[1] + f[24 + c] * w[2] + f[36 + c] * w[3];
    row[2 * c] = __float2half_rn(sinf(v));
    row[2 * c + 1] = __float2half_rn(cosf(v));
  }
  float dx = cx - means[3 * id], dy = cy - means[3 * id + 1], dz = cz - means[3 * id + 2];
  const float norm = fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), 1e-12f);
  dx /= norm; dy /= norm; dz /= norm;
  const float c0 = 0.28209479177387814f, c1 = 0.4886025119029199f;
  const float sh[9] = {c0, -c1 * dy, c1 * dz, -c1 * dx, 1.0925484305920792f * dx * dy, -1.0925484305920792f * dy * dz,
                       0.31539156525252005f * (2 * dz * dz - dx * dx - dy * dy), -1.0925484305920792f * dx * dz,
                       0.5462742152960396f * (dx * dx - dy * dy)};
  #pragma unroll
  for (int c = 0; c < 9; ++c) row[24 + c] = __float2half_rn(sh[c]);
  #pragma unroll
  for (int c = 33; c < kIn; ++c) row[c] = __float2half_rn(0.f);
}

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int SLOTS>   // abundance slots per lane: K <= 32 * SLOTS
__global__ void composite_kernel(const __half* __restrict__ y, const float* __restrict__ weight,
                                 const long* __restrict__ offsets, long pixels, int K,
                                 float* __restrict__ abundance, float* __restrict__ intensity_abundance,
                                 float* __restrict__ alpha) {
  const long p = (blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (p >= pixels) return;
  float acc[SLOTS], accs[SLOTS], total = 0.f;
  #pragma unroll
  for (int s = 0; s < SLOTS; ++s) acc[s] = accs[s] = 0.f;
  const int stride = K + 1;
  for (long j = offsets[p]; j < offsets[p + 1]; ++j) {
    const __half* row = y + j * stride;
    float logit[SLOTS], m = -INFINITY;
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
      const int k = lane + 32 * s;
      logit[s] = k < K ? __half2float(row[k]) : -INFINITY;
      m = fmaxf(m, logit[s]);
    }
    m = warp_max(m);
    float e[SLOTS], sum = 0.f;
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) { e[s] = lane + 32 * s < K ? expf(logit[s] - m) : 0.f; sum += e[s]; }
    sum = warp_sum(sum);
    const float raw = __half2float(row[K]);
    const float intensity = raw > 20.f ? raw : log1pf(expf(raw));     // softplus (beta 1, threshold 20)
    const float w = weight[j];
    total += w;
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
      const float a = e[s] / sum;
      acc[s] += w * a;
      accs[s] += (w * intensity) * a;
    }
  }
  const float scale = 1.f / fmaxf(total, 1e-8f);
  #pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const int k = lane + 32 * s;
    if (k < K) {
      abundance[p * K + k] = acc[s] * scale;
      intensity_abundance[p * K + k] = accs[s];
    }
  }
  if (lane == 0) alpha[p] = total;
}

// One layer for this warp's 16 rows: act[16, 16*KT] x W^T -> NT output tiles; bias (+ SiLU) in fp32. With `out` the
// result goes to global y (no activation), otherwise back into act (every k-tile is read before the first write).
template <int NT>
__device__ __forceinline__ void layer(__half* act, const __half* __restrict__ weights, int KT, const float* __restrict__ bias,
                                      float* scratch, __half* __restrict__ out, long first, long count, int width) {
  using namespace nvcuda;
  const int lane = threadIdx.x & 31;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NT];
  #pragma unroll
  for (int n = 0; n < NT; ++n) wmma::fill_fragment(acc[n], 0.f);
  for (int k = 0; k < KT; ++k) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::load_matrix_sync(a, act + k * 16, kLd);
    #pragma unroll
    for (int n = 0; n < NT; ++n) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::load_matrix_sync(b, weights + n * 16 * (16 * KT) + k * 16, 16 * KT);
      wmma::mma_sync(acc[n], a, b, acc[n]);
    }
  }
  __syncwarp();
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
    wmma::store_matrix_sync(scratch, acc[n], 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
      const int r = e >> 4, c = n * 16 + (e & 15);
      float v = scratch[e] + bias[c];
      if (out == nullptr) {
        act[r * kLd + c] = __float2half_rn(v / (1.f + __expf(-v)));
      } else if (first + r < count && c < width) {
        out[(first + r) * width + c] = __float2half_rn(v);
      }
    }
    __syncwarp();
  }
}

template <int OUT_TILES>
__global__ void __launch_bounds__(256) decode_kernel(
    const float* __restrict__ features, const float* __restrict__ means, const int* __restrict__ ids,
    const float* __restrict__ xyz, const long* __restrict__ kept, long count, float cx, float cy, float cz,
    const __half* __restrict__ w0, const float* __restrict__ b0, const __half* __restrict__ w1, const float* __restrict__ b1,
    const __half* __restrict__ w2, const float* __restrict__ b2, const __half* __restrict__ w3, const float* __restrict__ b3,
    int width, __half* __restrict__ y) {
  extern __shared__ __align__(32) unsigned char shared[];
  __half* act = reinterpret_cast<__half*>(shared);
  const int warp = threadIdx.x >> 5;
  float* scratch = reinterpret_cast<float*>(act + kHits * kLd) + warp * 256;
  const long base = blockIdx.x * (long)kHits;
  if (threadIdx.x < kHits) {
    const long j = base + threadIdx.x;
    __half* row = act + threadIdx.x * kLd;
    if (j < count) {
      const long h = kept[j];
      encode_row(features, means, ids[h], xyz[3 * h], xyz[3 * h + 1], xyz[3 * h + 2], cx, cy, cz, row);
    } else {
      for (int c = 0; c < kIn; ++c) row[c] = __float2half_rn(0.f);
    }
  }
  __syncthreads();                                   // from here on every warp works on its own 16 rows only
  __half* rows = act + warp * 16 * kLd;
  const long first = base + warp * 16;
  layer<8>(rows, w0, kIn / 16, b0, scratch, nullptr, first, count, width);
  layer<8>(rows, w1, 8, b1, scratch, nullptr, first, count, width);
  layer<8>(rows, w2, 8, b2, scratch, nullptr, first, count, width);
  layer<OUT_TILES>(rows, w3, 8, b3, scratch, y, first, count, width);
}

}  // namespace

torch::Tensor decode(torch::Tensor features, torch::Tensor means, torch::Tensor ids, torch::Tensor xyz, torch::Tensor kept,
                     double cx, double cy, double cz, torch::Tensor w0, torch::Tensor b0, torch::Tensor w1, torch::Tensor b1,
                     torch::Tensor w2, torch::Tensor b2, torch::Tensor w3, torch::Tensor b3, int64_t width) {
  TORCH_CHECK(features.is_cuda() && features.scalar_type() == torch::kFloat && features.size(1) == 48 && features.is_contiguous());
  TORCH_CHECK(means.scalar_type() == torch::kFloat && means.is_contiguous() && ids.scalar_type() == torch::kInt && ids.is_contiguous());
  TORCH_CHECK(xyz.scalar_type() == torch::kFloat && xyz.is_contiguous() && kept.scalar_type() == torch::kLong && kept.is_contiguous());
  for (const auto& w : {w0, w1, w2, w3}) TORCH_CHECK(w.scalar_type() == torch::kHalf && w.is_contiguous() && w.dim() == 2);
  TORCH_CHECK(w1.size(1) == 128 && w2.size(1) == 128 && w3.size(1) == 128);
  for (auto& b : {b0, b1, b2, b3}) TORCH_CHECK(b.scalar_type() == torch::kFloat && b.is_contiguous());
  TORCH_CHECK(w0.size(0) == 128 && w1.size(0) == 128 && w2.size(0) == 128 && w0.size(1) == kIn);
  const int tiles = w3.size(0) / 16;
  TORCH_CHECK(w3.size(0) % 16 == 0 && tiles >= 1 && tiles <= 8 && width <= w3.size(0) && b3.numel() == w3.size(0));
  const at::cuda::CUDAGuard guard(features.device());
  const long count = kept.numel();
  auto y = torch::empty({count, width}, features.options().dtype(torch::kHalf));
  if (count == 0) return y;
  const size_t bytes = kHits * kLd * sizeof(__half) + 8 * 256 * sizeof(float);
  const long blocks = (count + kHits - 1) / kHits;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto h = [](const torch::Tensor& t) { return reinterpret_cast<const __half*>(t.data_ptr<at::Half>()); };
#define LAUNCH(T) do { \
    C10_CUDA_CHECK(cudaFuncSetAttribute(decode_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes)); \
    decode_kernel<T><<<blocks, 256, bytes, stream>>>(features.data_ptr<float>(), means.data_ptr<float>(), ids.data_ptr<int>(), \
        xyz.data_ptr<float>(), kept.data_ptr<long>(), count, (float)cx, (float)cy, (float)cz, h(w0), b0.data_ptr<float>(), \
        h(w1), b1.data_ptr<float>(), h(w2), b2.data_ptr<float>(), h(w3), b3.data_ptr<float>(), (int)width, \
        reinterpret_cast<__half*>(y.data_ptr<at::Half>())); } while (0)
  switch (tiles) {
    case 1: LAUNCH(1); break; case 2: LAUNCH(2); break; case 3: LAUNCH(3); break; case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break; case 6: LAUNCH(6); break; case 7: LAUNCH(7); break; default: LAUNCH(8);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<torch::Tensor> composite(torch::Tensor y, torch::Tensor weight, torch::Tensor offsets) {
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kHalf && y.dim() == 2 && y.is_contiguous());
  TORCH_CHECK(weight.scalar_type() == torch::kFloat && weight.is_contiguous() && weight.numel() == y.size(0));
  TORCH_CHECK(offsets.scalar_type() == torch::kLong && offsets.is_contiguous());
  const at::cuda::CUDAGuard guard(y.device());
  const int K = y.size(1) - 1;
  TORCH_CHECK(K >= 1 && K <= 128, "1 <= K <= 128 materials");
  const long pixels = offsets.numel() - 1;
  auto options = weight.options();
  auto abundance = torch::empty({pixels, K}, options), accs = torch::empty({pixels, K}, options);
  auto alpha = torch::empty({pixels}, options);
  if (pixels == 0) return {abundance, accs, alpha};
  const int threads = 256;
  const long blocks = (pixels * 32 + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  const __half* yp = reinterpret_cast<const __half*>(y.data_ptr<at::Half>());
#define LAUNCH(S) composite_kernel<S><<<blocks, threads, 0, stream>>>(yp, weight.data_ptr<float>(), offsets.data_ptr<long>(), \
      pixels, K, abundance.data_ptr<float>(), accs.data_ptr<float>(), alpha.data_ptr<float>())
  if (K <= 32) LAUNCH(1); else if (K <= 64) LAUNCH(2); else LAUNCH(4);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {abundance, accs, alpha};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode, "fully fused per-hit encoding + fp16 MLP (tensor cores) -> raw outputs [M, K+1]");
  m.def("composite", &composite, "per-pixel softmax / softplus / blend-weighted compositing");
}

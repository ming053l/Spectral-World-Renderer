"""Abundance decoder, shared endmember dictionary, and the per-hit (pre-compositing) spectral renderer.

Data flow per hit (one Gaussian i intersected by a ray r; paper notation):
  hit lists (Hits.trace)       Gaussian id i, local hit position, visibility weight w_i(r)
  encode_local_features        harmonic features f_im (4 vertices x 12) -> local feature f_i(r) (barycentric
                               interpolation) -> gamma(f_i(r)) = [sin; cos] (24)
  directional_code             Y(d_i(r)): degree-2 real SH (9) of the direction from the Gaussian centre to the camera
  AbundanceField.decoder_outputs  z_i(r) = D_phi(gamma(f_i(r)), Y(d_i(r))) [K+1]: a 128x3 SiLU MLP whose first layer
                               also receives Y through a zero-initialised linear map (`camera_adapter`)
  AbundanceField.decode        a_i(r) = softmax(z[:K]), s_i(r) = softplus(z[K]), h_i(r) = s_i(r) E a_i(r)
  render_hits                  H(r) = sum_i w_i h_i,  A(r) = sum_i w_i a_i / sum_i w_i
The endmember dictionary E = [e_1..e_K], e_k = normalize(sigmoid(eta_k)), is stored as `global_endmember_logits`
(eta). Geometry is frozen; only the hit lists depend on it. A conditioning input of the decoder (for example a time
code) would enter where Y enters: `decoder_outputs`, its callers render_hits / render_view_fast, and the fused kernel.
"""
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from .defaults import Source

FEATURE_DIM = Source.feature_dim         # harmonic features per Gaussian: 4 tetrahedron vertices x 12
VERTEX_DIM = 12
ENCODED_DIM = 2 * VERTEX_DIM             # gamma(f) = [sin f; cos f]
DIRECTION_DIM = 9                        # degree-2 real SH
HIDDEN_DIM = 128


def barycentric_weights(xyz):
    """beta_im(r): barycentric coordinates of the local hit position in the virtual tetrahedron [M, 4]."""
    x, y, z = xyz.unbind(-1)
    return torch.stack((
        .25 + math.sqrt(6) / 12 * x - math.sqrt(2) / 12 * y - z / 12,
        .25 - math.sqrt(6) / 12 * x - math.sqrt(2) / 12 * y - z / 12,
        .25 + math.sqrt(2) / 6 * y - z / 12,
        .25 + z / 4), dim=-1)


def directional_code(direction):
    """Y(d): real SH basis through degree 2 of unit view directions [M, 3] -> [M, 9]."""
    x, y, z = direction.unbind(-1)
    c0, c1 = 0.28209479177387814, 0.4886025119029199
    c2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
    return torch.stack((
        torch.full_like(x, c0), -c1 * y, c1 * z, -c1 * x,
        c2[0] * x * y, c2[1] * y * z, c2[2] * (2 * z * z - x * x - y * y),
        c2[3] * x * z, c2[4] * (x * x - y * y)), dim=-1)


class AbundanceField(nn.Module):
    """Harmonic features f_im, the shared decoder D_phi and the endmember logits eta (state_dict keys: features,
    global_endmember_logits, network.*, camera_adapter.weight, mean_spectrum)."""

    def __init__(self, features, endmembers, mean_spectrum):
        super().__init__()
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f'Expected NHT features [N,{FEATURE_DIM}]')
        if endmembers.ndim != 2 or not torch.isfinite(endmembers).all():
            raise ValueError('Expected finite endmembers [K,B]')
        self.num_materials, self.num_bands = endmembers.shape
        self.features = nn.Parameter(features.detach().clone().float())
        e = endmembers.detach().to(self.features).clamp(1e-5, 1 - 1e-5)
        mean = mean_spectrum.detach().to(self.features).reshape(-1)
        if mean.shape != (self.num_bands,) or not torch.isfinite(mean).all():
            raise ValueError('Expected a finite training mean spectrum [B]')
        self.register_buffer('mean_spectrum', mean.clone())
        self.global_endmember_logits = nn.Parameter(torch.logit(e))
        self.network = nn.Sequential(                       # built on the CPU RNG, then moved
            nn.Linear(ENCODED_DIM, HIDDEN_DIM), nn.SiLU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(), nn.Linear(HIDDEN_DIM, self.num_materials + 1)).to(self.features)
        with torch.no_grad():                               # uniform abundances, intensity 1.5 at step 0
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()
            self.network[-1].bias[-1] = math.log(math.expm1(1.5))
        rng = torch.get_rng_state()                         # the zero adapter consumes no RNG
        self.camera_adapter = nn.Linear(DIRECTION_DIM, HIDDEN_DIM, bias=False).to(self.features)
        torch.set_rng_state(rng)
        nn.init.zeros_(self.camera_adapter.weight)
        self.register_buffer('primitive_means', torch.empty(0), persistent=False)

    def set_primitive_means(self, means):
        self.primitive_means = means.detach().to(self.features)

    def trainable_modules(self):
        return [('network', self.network), ('camera_adapter', self.camera_adapter)]

    def encode_local_features(self, ids, xyz):
        """gamma(f_i(r)) [M, 24] of hits on Gaussians `ids` at local positions `xyz`."""
        if xyz.shape != (ids.numel(), 3):
            raise ValueError('xyz must have shape [len(ids),3]')
        weights = barycentric_weights(xyz)
        f = (self.features[ids].reshape(-1, 4, VERTEX_DIM) * weights[..., None]).sum(1)
        return torch.stack((f.sin(), f.cos()), dim=-1).flatten(1)

    def endmembers(self):
        """The dictionary E [K, B]: unit-normalized sigmoid(eta_k)."""
        return F.normalize(self.global_endmember_logits.sigmoid(), dim=-1)

    def decoder_outputs(self, encoded, direction_code):
        """z = D_phi(gamma(f), Y(d)) [M, K+1]: K abundance logits and one intensity pre-activation."""
        if direction_code is None or direction_code.shape != (len(encoded), DIRECTION_DIM):
            raise ValueError(f'Expected the degree-2 directional SH code [N,{DIRECTION_DIM}]')
        h = self.network[0](encoded) + self.camera_adapter(direction_code)
        for layer in self.network[1:-1]:
            h = layer(h)
        return self.network[-1](h)

    def decode(self, encoded, direction_code=None):
        """Per hit: abundances a [M, K], intensity s [M, 1] and spectrum h = s E a [M, B]."""
        z = self.decoder_outputs(encoded, direction_code)
        abundance = z[:, :-1].softmax(-1)
        intensity = F.softplus(z[:, -1:])
        e = self.endmembers()
        weighted = abundance / e.square().sum(-1).clamp_min(1e-12).sqrt()
        return {'spectrum': intensity * (weighted @ e), 'abundance': abundance, 'intensity': intensity}

    # earlier names, still used by analysis scripts outside the package
    encoded_features = encode_local_features
    global_endmembers = endmembers

    def parameter_counts(self):
        return {'total': sum(p.numel() for p in self.parameters()), 'features': self.features.numel(),
                'network': sum(p.numel() for p in self.network.parameters()),
                'endmembers': self.global_endmember_logits.numel(), 'camera_adapter': self.camera_adapter.weight.numel(),
                'semantic_adapter': 0, 'frozen_semantic_code_values': 0}


GEOMETRY = ('means', 'scales', 'quats', 'opacities')


def load_model(path, device='cuda'):
    """Trained decoder (model_ema.pt) and its splats (frozen geometry + trained features)."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    splats = {k: saved[k].float().to(device) for k in GEOMETRY}
    splats['features'] = state['features'].float().to(device)
    model = AbundanceField(state['features'], saved['initial_endmembers'][:state['global_endmember_logits'].shape[0]],
                           saved['mean_spectrum']).to(device)
    model.load_state_dict(state, strict=True)
    model.set_primitive_means(splats['means'])
    model.eval()
    return model, splats


def _gsplat():
    """The vendored hit-trace gsplat (third_party/gsplat): its CUDA library, rasterizer and NHT parameters."""
    from . import third_party
    third_party.use()
    from gsplat.cuda._backend import _C
    from gsplat.rendering import rasterization
    from gsplat.nht import NHTParams
    return _C, rasterization, NHTParams


class Hits:
    """Per-view hit lists (Gaussian id, local hit position, blend weight per pixel) of the frozen field.

    Traced on the GPU with the hit-trace build of the NHT gsplat kernel (two passes: count, record). `preload` traces
    a set of views once and keeps them resident: on the GPU up to `budget_gib`, the rest in host memory, from where
    `select` gathers the hits of the requested pixels before copying them to the GPU. Views that were not preloaded
    are traced when used and kept on the GPU while the budget allows.
    """

    def __init__(self, splats, pack, budget_gib, margin_gib=2.):
        self.s = splats
        self.margin = margin_gib * 2 ** 30              # free GPU memory left to other processes when caching
        self.pack = pack
        self.budget = budget_gib * 2 ** 30
        self.used = 0
        self.cache = {}
        self.host = {}
        self.traces = 0
        self.trace_seconds = 0.
        self.hits_per_pixel = {}

    @staticmethod
    def nbytes(hits):
        return sum(v.numel() * v.element_size() for v in hits.values())

    def preload(self, views):
        """Trace every view once; GPU-resident within the budget, host-resident (pinned) beyond it."""
        for i in views:
            if i in self.cache or i in self.host:
                continue
            hits = self.trace(i)
            size = self.nbytes(hits)
            if self.used + size <= self.budget and self._room(size):
                self.cache[i] = hits
                self.used += size
            else:
                # copied straight into pinned memory: one transfer instead of a pageable copy plus pinning
                self.host[i] = {k: torch.empty(v.shape, dtype=v.dtype, pin_memory=True).copy_(v) for k, v in hits.items()}
            del hits

    def _room(self, size):
        """The card still has `size` bytes plus the margin free (shared GPUs; only placement depends on it)."""
        return torch.cuda.mem_get_info()[0] >= size + self.margin

    def select(self, i, pixels):
        """Hits of the given flattened pixels (a GPU tensor) of view i and, per hit, its pixel index in `pixels`."""
        if i in self.host:
            hits, ids = select_pixels(self.host[i], pixels.cpu())
            return {k: v.to(pixels.device, non_blocking=True) for k, v in hits.items()}, ids.to(pixels.device, non_blocking=True)
        return select_pixels(self.get(i), pixels)

    @torch.no_grad()
    def trace(self, i):
        _C, rasterization, NHTParams = _gsplat()
        tic = time.perf_counter()
        s, dev = self.s, self.s['means'].device
        h, w = self.pack.height, self.pack.width
        n = h * w
        viewmat = torch.linalg.inv(self.pack.c2w(i).to(dev))[None]
        K = self.pack.K(i).to(dev)[None]
        counts = torch.zeros(n, dtype=torch.int32, device=dev)
        offsets = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        ids = torch.empty(0, dtype=torch.int32, device=dev)
        xyz = torch.empty(0, 3, device=dev)
        weight = torch.empty(0, device=dev)
        flag = torch.zeros(1, dtype=torch.int32, device=dev)

        def render():
            rasterization(means=s['means'], quats=s['quats'], scales=s['scales'].exp(), opacities=s['opacities'].sigmoid(),
                          colors=s['features'], viewmats=viewmat, Ks=K, width=w, height=h, packed=False, with_ut=True,
                          with_eval3d=True, tile_size=8, nht_params=NHTParams(enabled=True, ray_dir_scale=3.))
            torch.cuda.synchronize()
            assert int(flag) == 0, int(flag)
        failed = False
        try:
            _C.set_nht_hit_trace(1, counts, offsets, ids, xyz, weight, flag)
            render()
            expected = counts.clone()
            offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), expected.long().cumsum(0)])
            total = int(offsets[-1])
            ids = torch.empty(total, dtype=torch.int32, device=dev)
            xyz = torch.empty(total, 3, device=dev)
            weight = torch.empty(total, device=dev)
            counts.zero_()
            _C.set_nht_hit_trace(2, counts, offsets, ids, xyz, weight, flag)
            render()
            assert torch.equal(counts, expected), 'hit count changed between the count and record passes'
        except BaseException:
            failed = True
            raise
        finally:
            try:                                       # switch tracing off; never mask the original error
                _C.set_nht_hit_trace(0, counts, offsets, ids, xyz, weight, flag)
            except Exception:
                if not failed:
                    raise
        self.traces += 1
        self.trace_seconds += time.perf_counter() - tic
        self.hits_per_pixel[i] = total / n
        return {'offsets': offsets, 'ids': ids, 'xyz': xyz, 'weight': weight}

    def get(self, i, keep=True):
        if i in self.cache:
            return self.cache[i]
        if i in self.host:
            return {k: v.to(self.s['means'].device) for k, v in self.host[i].items()}
        hits = self.trace(i)
        size = self.nbytes(hits)
        if keep and self.used + size <= self.budget:
            self.cache[i] = hits
            self.used += size
        return hits


def select_pixels(hits, pixels):
    """Hits of the given pixels (flattened) and, per hit, the index of its pixel in `pixels`."""
    offsets = hits['offsets']
    starts = offsets[pixels]
    lengths = offsets[pixels + 1] - starts
    device = pixels.device
    total = int(lengths.sum())                         # the one host sync; repeat_interleave then needs none
    output_ids = torch.repeat_interleave(torch.arange(len(pixels), device=device), lengths, output_size=total)
    base = torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths, output_size=total)
    positions = torch.arange(total, device=device) - base + torch.repeat_interleave(starts, lengths, output_size=total)
    return {k: hits[k][positions] for k in ('ids', 'xyz', 'weight')}, output_ids


def render_hits(model, hits, pixel_ids, num_pixels, camera_center, chunk=1 << 18):
    """Decode every hit, composite with the blend weights: spectrum [P, B], alpha [P, 1], abundance [P, K]."""
    means = model.primitive_means
    alpha = model.features.new_zeros(num_pixels, 1)
    hsi = model.features.new_zeros(num_pixels, model.num_bands)
    abundance = model.features.new_zeros(num_pixels, model.num_materials)
    for start in range(0, len(hits['ids']), chunk):
        sl = slice(start, start + chunk)
        ids = hits['ids'][sl].long()
        xyz = hits['xyz'][sl].float()
        weight = hits['weight'][sl].float().reshape(-1, 1)
        pix = pixel_ids[sl].long()
        code = directional_code(F.normalize(camera_center[None] - means[ids], dim=-1))
        alpha = alpha.index_add(0, pix, weight)
        out = model.decode(model.encode_local_features(ids, xyz), code)
        hsi = hsi.index_add(0, pix, weight * out['spectrum'])
        abundance = abundance.index_add(0, pix, weight * out['abundance'])
    return {'hsi': hsi, 'alpha': alpha, 'abundance': abundance / alpha.clamp_min(1e-8)}


@torch.no_grad()
def render_view(model, hit_source, i, center, keep=False, pixel_chunk=65536):
    """Reference full-image render of view i (FP32): spectrum [H*W, B], abundance [H*W, K], alpha [H*W, 1]."""
    hits = hit_source.get(i, keep=keep)
    n = hit_source.pack.height * hit_source.pack.width
    device = model.features.device
    hsi = torch.empty(n, model.num_bands, device=device)
    abundance = torch.empty(n, model.num_materials, device=device)
    alpha = torch.empty(n, 1, device=device)
    for start in range(0, n, pixel_chunk):
        pix = torch.arange(start, min(n, start + pixel_chunk), device=device)
        sub, ids = select_pixels(hits, pix)
        out = render_hits(model, sub, ids, len(pix), center)
        hsi[pix], abundance[pix], alpha[pix] = out['hsi'], out['abundance'], out['alpha']
    return hsi, abundance, alpha


@torch.no_grad()
def render_view_fast(model, hit_source, i, center, tau=1e-4, chunk=1 << 20, keep=False, mixing=False):
    """Fast render of view i: hits with blend weight < tau skipped, MLP in fp16, endmember mixing after compositing.

    Mixing after compositing is exact (the spectrum is linear in intensity * abundance); fp16 and the skip change the
    paper's TEST scores by < 1e-3 (PSNR / SAM / mIoU). Returns the same three maps as render_view; with mixing=True
    also the composited coefficients sum_i w_i s_i a_i [H*W, K] (spectrum = coefficients @ E, e.g. with endmembers
    switched off in the viewer).
    """
    n = hit_source.pack.height * hit_source.pack.width
    device = model.features.device
    sub, pix_all = select_pixels(hit_source.get(i, keep=keep), torch.arange(n, device=device))
    w_all = sub['weight'].float()
    keep_hit = w_all >= tau
    ids_all, xyz_all, w_all, pix_all = sub['ids'][keep_hit], sub['xyz'][keep_hit], w_all[keep_hit], pix_all[keep_hit]
    K = model.num_materials
    acc = torch.zeros(n, K, device=device)                 # sum w a
    accs = torch.zeros(n, K, device=device)                # sum w s a
    alpha = torch.zeros(n, device=device)
    for st in range(0, len(ids_all), chunk):
        sl = slice(st, st + chunk)
        ids, w, pix = ids_all[sl].long(), w_all[sl], pix_all[sl].long()
        code = directional_code(F.normalize(center[None] - model.primitive_means[ids], dim=-1))
        with torch.autocast('cuda', dtype=torch.float16):
            y = model.decoder_outputs(model.encode_local_features(ids, xyz_all[sl].float()), code).float()
        a, s = y[:, :-1].softmax(-1), F.softplus(y[:, -1:])
        alpha.index_add_(0, pix, w)
        acc.index_add_(0, pix, w[:, None] * a)
        accs.index_add_(0, pix, (w[:, None] * s) * a)
    e = model.endmembers()
    hsi = (accs / e.square().sum(-1).clamp_min(1e-12).sqrt()) @ e
    if mixing:
        return hsi, acc / alpha.clamp_min(1e-8)[:, None], alpha[:, None], accs
    return hsi, acc / alpha.clamp_min(1e-8)[:, None], alpha[:, None]


_FUSED_INPUT_DIM = 48                    # kIn of csrc/fused_render.cu


@torch.no_grad()
def render_view_fused(model, hit_source, i, center, tau=1e-4, mixing=False):
    """render_view_fast with fused CUDA kernels (abundance_gs/csrc/fused_render.cu): one kernel decodes the kept hits
    (encoding + the whole fp16 MLP on tensor cores), one composites per pixel (softmax, softplus, blend-weighted sums
    in hit order). Same skip rule and post-compositing mixing; agrees with render_view_fast to fp16 rounding (the
    first layer and the camera adapter are one product, SiLU inputs are not rounded to fp16, the sums use no atomics).
    mixing=True: as in render_view_fast.
    """
    from . import ext
    kernels = ext.load('abgs_render')
    hits = hit_source.get(i, keep=False)
    weight = hits['weight']
    keep = weight >= tau
    kept = keep.nonzero().squeeze(1)
    kept_before = torch.cat([keep.new_zeros(1, dtype=torch.int64), keep.cumsum(0)])
    offsets = kept_before[hits['offsets']].contiguous()         # per-pixel ranges of the kept hits
    first, last = model.network[0], model.network[-1]
    hidden = [m for m in model.network[2:-1] if isinstance(m, nn.Linear)]
    # the kernel's first layer: [gamma(f) 24 | Y(d) 9 | zero padding to 3 tensor-core k-tiles]
    w0 = torch.zeros(first.out_features, _FUSED_INPUT_DIM, device=weight.device)
    w0[:, :ENCODED_DIM], w0[:, ENCODED_DIM:ENCODED_DIM + DIRECTION_DIM] = first.weight, model.camera_adapter.weight
    rows = -(-last.out_features // 16) * 16
    w3 = torch.zeros(rows, last.in_features, device=weight.device)
    b3 = torch.zeros(rows, device=weight.device)
    w3[:last.out_features], b3[:last.out_features] = last.weight, last.bias
    y = kernels.decode(model.features.detach().float().contiguous(), model.primitive_means.float().contiguous(),
                       hits['ids'].contiguous(), hits['xyz'].float().contiguous(), kept, *center.float().tolist(),
                       w0.half(), first.bias.float().contiguous(), hidden[0].weight.half().contiguous(),
                       hidden[0].bias.float().contiguous(), hidden[1].weight.half().contiguous(),
                       hidden[1].bias.float().contiguous(), w3.half(), b3, last.out_features)
    abundance, accs, alpha = kernels.composite(y, weight[kept].float().contiguous(), offsets)
    e = model.endmembers()
    hsi = (accs / e.square().sum(-1).clamp_min(1e-12).sqrt()) @ e
    if mixing:
        return hsi, abundance, alpha[:, None], accs
    return hsi, abundance, alpha[:, None]


sh2 = directional_code                   # earlier name, still used by analysis scripts outside the package

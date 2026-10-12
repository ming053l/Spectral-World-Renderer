# Third-party code

Only the code that Abundance-GS executes is included. Upstream licence headers are unchanged; every modified line is
marked `Abundance-GS:`. CUDA libraries are compiled from these sources by `abundance_gs/ext.py` into the build
directory (`python abgs.py build`).

| directory | origin | version | licence | changes |
|---|---|---|---|---|
| `gsplat/` | gsplat (nerfstudio-project) with the NHT extensions of neural-harmonic-textures (NVIDIA), plus a hit-trace mode of the NHT rasterization kernel (`csrc/RasterizeToPixelsFromWorldNHT3DGSFwd.cu`, `ext.cpp`: `set_nht_hit_trace`) | neural-harmonic-textures 2ae7ab0a, gsplat 068dc659 (1.5.3) | Apache-2.0 (`gsplat/LICENSE`) | `cuda/_backend.py` loads the library through ext.py (`cuda/build.py` removed); unused Python subpackages removed; `csrc/third_party/glm` reduced to its headers |
| `fused_ssim/` | fused-ssim (R. Goel), used by the anchor and the source trainer | a7c48d6d | MIT | CUDA backend only; library via ext.py (`fused_ssim_cuda`) |
| `fastgs/` | FastGS trainer, with a `--seed` option, a fixed test-view list and a Blender camera reader patch | FastGS 44e02a5c | MIT (`LICENSE`), Inria 3DGS licence (`LICENSE_ORIGINAL.md`) | `train.py`: viewer, tensorboard, LPIPS and the unused report function removed; `gaussian_model.py`: only the default Adam; `distCUDA2` from ext.py; rasterizer / SSIM imports point to the packages here |
| `fastgs_rasterizer/` | `diff-gaussian-rasterization_fastgs` (FastGS submodule) | FastGS 44e02a5c | Inria 3DGS licence (`LICENSE.md`) | `_C` from ext.py; `#include <cstdint>` added to `rasterizer_impl.h` (g++ ≥ 13) |
| `simple_knn/` | `simple-knn` (FastGS submodule) | FastGS 44e02a5c | Inria 3DGS licence | sources only; `distCUDA2` from ext.py; `#include <cstdint>` added (g++ ≥ 13) |
| `gs_rasterizer/` | `diff-gaussian-rasterization` (graphdeco-inria), version with antialiasing and inverse-depth output; kept next to the FastGS rasterizer because the anchor's RGB score must use the standard 3DGS culling (the FastGS rasterizer's compact boxes render differently) | — | Inria 3DGS licence | `_C` from ext.py |

Adapted into the package:

- `abundance_gs/scene_loader.py`: camera / image / HSI loading from the 3D Gaussian Splatting scene loader
  (graphdeco-inria; `scene/colmap_loader.py`, `scene/dataset_readers.py`, `scene/cameras.py`, `utils/camera_utils.py`,
  `utils/graphics_utils.py`, `utils/general_utils.py`; Inria 3DGS licence).
- `abundance_gs/nht/trainer.py`: the code paths of neural-harmonic-textures `aov/examples/simple_trainer_nht_aov.py`
  (2ae7ab0a, Apache-2.0) that stage `source` uses.

Environment packages (built by `env/setup.sh`, pinned in `env/requirements.lock`): PyTorch 2.9.1+cu126, torchvision,
NumPy, SciPy, torchmetrics, matplotlib, imageio, plyfile, Pillow, and tiny-cuda-nn (NVlabs) built from commit
0109538c.

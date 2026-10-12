<div align="center">

# Spectral World Renderer

### Towards Verifiable 3D Hyperspectral Unmixing

**Chia-Ming Lee<sup>1,2</sup> · Yu-Jou Hsiao<sup>3</sup> · Ming-Ching Chang<sup>1</sup> · Xin Li<sup>1</sup> · Yu-Lun Liu<sup>2</sup> · Chih-Chung Hsu<sup>2,3</sup>**

<sup>1</sup>University at Albany, State University of New York · <sup>2</sup>National Yang Ming Chiao Tung University · <sup>3</sup>National Cheng Kung University

[![Project page](https://img.shields.io/badge/Project-Page-0B7A75?style=for-the-badge)](https://ming053l.github.io/Spectral-World-Renderer/)
[![Paper](https://img.shields.io/badge/Paper-PDF%20%2B%20supplement-B31B1B?style=for-the-badge)](https://ming053l.github.io/Spectral-World-Renderer/assets/paper.pdf)
[![Video](https://img.shields.io/badge/Video-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=5eKuSb7DITo)
![Code](https://img.shields.io/badge/Code-coming%20soon-6E7781?style=for-the-badge)

**See what a 3D scene is made of, not only how it looks.**

<img src=".github/readme/query_modes.webp" width="560" alt="Material queries in Market, Karst, City and Botanical, each lit in its own colour: terracotta by name, wood by a click in shadow, stone or travertine, limestone by a click in shadow, water, a living-leaf reference spectrum, terracotta, a near-infrared index and wood.">

</div>

## Overview

Cameras and RGB-based 3D models record colour, so surfaces of the same colour but different composition, such as a green-painted panel and a leaf, look alike. Hyperspectral images separate them, but scenes with known material composition are rare, so a 3D decomposition is hard to verify.

- **Spectral World Renderer** generates indoor and outdoor hyperspectral worlds whose material composition is known by construction. A coding agent turns reference images and text into executable scene programs; measured spectra and shared sub-pixel rendering samples then yield aligned hyperspectral observations, material labels and area proportions.
- **SWR-Bench** comprises eight indoor and outdoor worlds with material and illumination variation, rendered with 64 bands from 420 to 980 nm.
- **Abundance-GS** decodes abundances over a shared endmember dictionary at every ray intersection of a compact Gaussian representation, so every primitive carries queryable abundances. A multi-view appearance regularization assigns endmembers to distinct appearances without material labels.

## Query any material

Abundances and spectra are stored on explicit, rasterizable Gaussians, so a query is a selection over primitives, consistent across views and free of labels or per-image classifiers.

| Query mode | Example | What it shows |
|---|---|---|
| Material name | `terracotta`, `water`, `wood` | the dominant endmember of each surface |
| Click | a shaded pixel on wood or limestone | the sunlit parts of the same material are retrieved too: 77 % vs 10 % for an RGB colour query (mean over four worlds) |
| Reference spectrum | a living-leaf reflectance from a spectral library | foliage is separated from foliage-coloured minerals (2° apart in RGB, 42° in spectrum) |
| Band index | near-infrared index | materials beyond the visible range |
| Combination | `stone ∨ travertine` | several materials at once |

A spectral query takes about 3 ms.

## Results at a glance

| | Abundance-GS | Reference |
|---|---|---|
| Material identification on SWR-Bench | best or on par in all eight worlds | Table 2 |
| City training, full pipeline | **6 min 37 s** | UnMix-NeRF 24 min 17 s · 3DGS+VCA/NNLS 85 min 2 s |
| Rendering with spectra and label maps (960×540×64) | **34.9 views/s**, with our optimized fused fp16 decoder | UnMix-NeRF 0.18 · 3DGS+VCA/NNLS 3.6 |
| Real captures (NeSpoF, HSI-SC, HS-NeRF) | highest PSNR on each source | Table 5 |
| Mineral-leaf query in City | no ordinary tree retrieved | RGB query: eleven ordinary trees |

## Citation

If you find this work useful, please cite:

```bibtex
@article{lee2026swr,
  title   = {Spectral World Renderer: Towards Verifiable 3D Hyperspectral Unmixing},
  author  = {Lee, Chia-Ming and Hsiao, Yu-Jou and Chang, Ming-Ching and
             Li, Xin and Liu, Yu-Lun and Hsu, Chih-Chung},
  journal = {Preprint, under review},
  year    = {2026},
  url     = {https://ming053l.github.io/Spectral-World-Renderer/}
}
```

## Repository contents

This repository hosts the project page, the paper with its supplementary material, and rendered clips. Research code and datasets are not part of this release.

```
docs/
  index.html, style.css, app.js   static page, no build step
  assets/paper.pdf                 paper and supplementary material
  assets/fig/                      paper figures rasterized for the web (WebP)
  assets/video/                    poster frame of the overview video (hosted on YouTube)
  assets/query/                    material-query clips (720p) per world and query
  assets/tetra/                    bounding-tetrahedra clips (720p)
  assets/worlds/                   SWR-Bench world thumbnails
.github/
  workflows/pages.yml              GitHub Pages deployment
  readme/                          README preview
```

## Local preview

```bash
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>. Python's built-in server does not answer HTTP range requests, so video seeking may be unreliable locally; GitHub Pages serves ranges. GitHub Pages deploys `docs/` through `.github/workflows/pages.yml` on every push to `main` that changes `docs/`; in **Settings → Pages**, the source must be **GitHub Actions**.

## Acknowledgements

Website layout adapted from the [Nerfies](https://nerfies.github.io/) template, as used by [LongSplat](https://linjohnss.github.io/longsplat/).

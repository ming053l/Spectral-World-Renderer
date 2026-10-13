# SWR

Code for *Spectral World Renderer: 3D Hyperspectral Unmixing*. The Python package `abundance_gs`
and the `abgs` command implement the unmixing representation of SWR (Spectral World Renderer).

[Chia-Ming Lee](https://ming053l.github.io/)<sup>1,2</sup> · Yu-Jou Hsiao<sup>3</sup> · [Ming-Ching Chang](https://www.albany.edu/faculty/mchang2/)<sup>1</sup> · [Xin Li](https://scholar.google.com/citations?user=gMBvzGoAAAAJ&hl=en)<sup>1</sup> · [Yu-Lun Liu](https://yulunalexliu.github.io/)<sup>2</sup> · [Chih-Chung Hsu](https://cchsu.info/)<sup>2,3</sup>

<sup>1</sup>University at Albany, State University of New York · <sup>2</sup>National Yang Ming Chiao Tung University · <sup>3</sup>National Cheng Kung University

[Project page](https://ming053l.github.io/Spectral-World-Renderer/) · [Paper + supplement (PDF)](https://ming053l.github.io/Spectral-World-Renderer/assets/paper.pdf) · [Video](https://www.youtube.com/watch?v=3FrSH7MaSjU)

An RGB Gaussian-splatting anchor gives the scene geometry. A harmonic-texture feature field is fitted on the anchor
from the RGB images. A per-hit decoder turns every Gaussian intersection along a camera ray into K abundances over a
shared endmember dictionary plus an intensity; spectra and abundance maps are obtained by alpha compositing. The
multi-view appearance regularization groups the pooled training pixels into appearance clusters by spectral shape and
brightness: cluster centre k initializes endmember k, and the persistent assignment regularization L_p ties the
composited abundances to the cluster labels y_r (weight .05, fading to a floor of .005 over 3k updates).

## Install

```bash
env/setup.sh                 # conda env in ~/envs/abundance_gs (or $ABGS_ENV_PREFIX): python 3.10, CUDA 12.9 compiler,
                             # pinned packages (env/requirements.lock), tiny-cuda-nn, LPIPS weights; then
                             # `pip install -e .` (console script `abgs`) and `abgs build` (all CUDA extensions)
env/run.sh -1 tests/test_package.py
```

In an existing environment with the pinned packages, the same two steps by hand:

```bash
pip install --no-deps --no-build-isolation -e .
abgs build                   # compiles the CUDA extensions (gsplat, rasterizers, SSIM, fused renderer), ~10-15 min
```

`abgs <stage> ...` (or `python abgs.py <stage> ...` from any `python3`) re-executes itself in the environment of
`env/env.sh`, which also selects the compiler: on hosts whose system g++ is older than 11, the environment's
conda g++ builds the extensions. Extensions live in `$ABGS_BUILD_DIR` (default `~/.cache/abundance_gs/build`) and
are rebuilt only when their sources change; an extension missing at first use is built then. `--gpu N` selects the
card; `--gpu auto` (default) takes the card with the most free memory.

## Data roots

Scene configurations refer to two roots, read from the environment:

```
$ABGS_DATA_ROOT/
  swr_bench/<scene>/{noisy,clean,anchor_input}     SWR-Bench datasets and the RGB input of the anchor
  {nespof,hsi_sc,hs_nerf}/<scene>/{anchor_input,source}
  packs/<scene>/{train,eval}                       packs (cameras, split, RGB, HSI paths) for training and evaluation
$ABGS_MODELS_ROOT/<scene>/
  anchor/point_cloud.ply  source/ckpt.pt  clusters_K<K>/  decoder/model_ema.pt  eval/  reference/{state.pt,anchor}
```

Alternatively, an untracked `configs/local_roots.json` maps the logical paths to existing locations:
`{"roots": {"ABGS_DATA_ROOT": "/data/abgs"}, "links": {"${ABGS_MODELS_ROOT}/market/decoder": "/runs/m/decoder"}}`
(`links`: longest matching prefix; environment variables take precedence).

## Stages

| stage | does | output |
|---|---|---|
| `anchor` | FastGS 3DGS, multi-view density control, 30k iterations | `point_cloud/iteration_30000/point_cloud.ply`, `anchor.json` |
| `pack` | cameras, split, RGB and HSI paths of a scene | `pack.json`, `rgb.npy` |
| `source` | anchor Gaussians -> harmonic-texture feature field, 5k RGB steps | `ckpts/ckpt_4999_rank0.pt`, `run.json` |
| `prepare` | appearance clusters of the TRAIN pixels | cache: `initial_endmembers.npy`, `sampling_bins/`, `dense_labels/`, `preparation.json` |
| `train` | abundance decoder, 30k updates | `model_ema.pt`, `status.json`, `protocol.json`, `trace.json` |
| `eval` | render a pack, spectral metrics, abundance maps, material scorer | `spectral_metrics.json`, `predictions/`, `benchmark_material_evaluation.json` |
| `render` | spectra and abundance maps of chosen views | `<view>.npy`, `<view>_abundance.npz` |
| `fps` | render speed over the TEST views (reference, fast and fused renderer) | json |
| `run` | `[pack] anchor source prepare train [eval]` for one scene; completed stages are skipped | `WORKDIR/...` |
| `scenes`, `build` | list scene configurations; compile CUDA extensions | |

```bash
abgs anchor  --scene-dir $D/anchor_input --resolution 1 --output $W/anchor
abgs pack    --state $M/reference/state.pt --scene-id market --output $W/pack
abgs source  --pack $P/train --anchor-ply $W/anchor/point_cloud/iteration_30000/point_cloud.ply --output $W/source
abgs prepare --pack $P/train --materials 24 --output $W/clusters_K24
abgs train   --pack $P/train --source $W/source/ckpts/ckpt_4999_rank0.pt --clusters $W/clusters_K24 \
             --materials 24 --output $W/decoder
abgs eval    --model $W/decoder/model_ema.pt --pack $P/eval --clean-dataset $D/clean --scorer benchmark --output $W/eval
abgs render  --model $W/decoder/model_ema.pt --pack $P/eval --views test --fast --output $W/maps
abgs fps     --model $W/decoder/model_ema.pt --pack $P/eval --output $W/fps.json
abgs run     --scene market --workdir $W
```

## Scene configurations

`configs/scenes/<scene>.json` (on top of `configs/datasets/<dataset>.json`) holds the data, packs, K, preset,
evaluation settings, figure views and the trained models of each scene in the paper. With `--scene NAME`, every
option not given on the command line is taken from it; `--print-config` shows the result.

```bash
abgs scenes
abgs eval   --scene market --output $W/eval
abgs render --scene karst --views figures --fast --output $W/maps
abgs train  --scene market -K 16 --output $W/k16
```

Scenes: SWR-Bench market, botanical, study, office, karst, forbidden_city, nvidia_hq, city and the City variants
city_metamer, city_stress, city_illum_am, city_illum_pm; real captures nespof_scene1-4, apple, basil, rosemary,
harsh_set1, harsh_set2, fakeplant1_2.

| field | meaning |
|---|---|
| `data.anchor_scene`, `data.noisy_dataset`, `data.clean_dataset`, `data.clean_hsi_subdir` | anchor input, noisy observations, clean references |
| `data.label_space` | label space of the scorer report that holds the paper's mIoU (informational) |
| `packs.train`, `packs.eval` | training pack (SWR-Bench: 480x270) and evaluation pack (SWR-Bench: native 960x540) |
| `decoder.preset`, `decoder.materials`, `decoder.val_every_n_train` | preset, K, held-out HSI TRAIN views |
| `eval.scorer`, `eval.scorer_args`, `eval.foreground` | material scorer (`abundance_gs/scorers/<name>.py`), its command line, GT foreground rule |
| `views.figures`, `views.retain` | figure views (`render --views figures`); abundance maps kept after scoring |
| `paper_model.*` | anchor ply, source checkpoint, cluster cache, decoder, evaluation of the paper's model |

## Presets and options of `train`

| | `standard` (default) | `reference` (`--reference`) |
|---|---|---|
| endmember initialization `--init` | appearance-cluster centres (`clusters`) | angular farthest-point TRAIN spectra (`fps`) |
| L_p (`--no-regularization`) | on: .05 fading over 3k updates, floor `--reg-floor` .005 | off |
| dictionary frozen `--freeze-endmembers` | 3,000 updates | 300 updates |

Other options: `--materials/-K` (16; the paper uses 24 for Market, Botanical, Study, Office, 16 for Karst and the
real captures, 32 for Forbidden City, NVIDIA HQ and City), `--steps` 30000, `--pixels` 2048 per update, `--seed` 42,
`--w-sam` .05 (loss = L1 + w_sam SAM + L_p), `--clusters` (cache from `prepare`; built at `OUTPUT/appearance_clusters`
if absent), `--train-max-views` / `--val-every-n-train` (HSI-supervision subset; `prepare` must use the same values),
`--eval-steps` (TEST curve), `--keep-cubes`, `--hit-cache-gib` (the hit lists of all TRAIN views are traced once at
the start; up to this many GiB stay on the GPU, the rest in pinned host memory; default `auto`: the card's free
memory minus a 2 GiB margin and a 3 GiB working set, with `--gpu-memory-budget-mib` raised to match unless given;
a fixed value must fit the budget with the working set). Where the hit lists live changes speed only, not results.
Cluster caches are validated (K, bands, grid, pack, TRAIN subset, hashes) before training and never overwritten.

`eval`: `--clean-dataset` selects the benchmark layout (otherwise TEST views are scored against the pack's HSI),
`--foreground {coverage,surface}`, `--class-spectral`, `--fast` (fp16 MLP, hits with weight < 1e-4 skipped; scores
within 1e-3), `--fused` (the fast renderer with fused CUDA kernels, `abundance_gs/csrc/fused_render.cu`: per-hit
encoding and the fp16 MLP on tensor cores in one kernel, compositing in another; agrees with `--fast` to fp16
rounding), `--scorer/--scorer-args/--dataset`, `--retain-views`. `prepare --device cuda` clusters on the GPU (not
bitwise equal to the CPU path; recorded in `preparation.json`).

## Outputs

Every stage writes `abgs_record.json` into its output directory: the stage, digests of its inputs, the options
that determine the result, and its main outputs (`abundance_gs/records.py`). `run` skips a stage whose record
matches the current inputs and options; output directories without a record are read through their stage files.

- `status.json`: TEST metrics of the raw and EMA decoder at the pack resolution (`unified_hsi_metrics`, macro mean
  over views), training time, FPS, model size; `protocol.json`: configuration, inputs and the SHA256 of the sampled
  view/pixel sequence; `model_ema.pt`: geometry, features, decoder and endmembers.
- `spectral_metrics.json`: PSNR / SAM / p95 SAM against the clean and noisy cubes, GT-foreground and unclipped
  variants, noise floor, TRAIN metrics; `benchmark_material_evaluation.json`: TRAIN-calibrated many-to-one
  endmember-to-material mapping, TEST accuracy, mIoU and per-class IoU.

## Extending

Module boundaries:

- Command line: `cli.py` (options and the stage registry `STAGES`), `entry.py` (environment, GPU choice, dispatch),
  `scenes.py` (scene configurations), `records.py` (stage records), `pipeline.py` (stage `run`).
- Data: `data.py` (`Pack`: the views of a scene, addressed by index), `export.py` and `scene_loader.py` (pack export).
- Stages: `anchor.py`; `source.py` and `nht/` (harmonic features); `clusters.py` and `clusters_gpu.py`; `train.py`;
  `evaluate.py`, `render.py`, `fps.py`.
- Model: `model.py` (decoder, dictionary, hit lists, the three renderers), `inference.py` (whole-view rendering
  helpers), `metrics.py`, `scorers/` (material scorers, listed in `scorers.SCORERS`).

**A scene** is a file `configs/scenes/<name>.json` with `"extends": "<dataset>"`; it is merged key by key onto
`configs/datasets/<dataset>.json`, and `${ABGS_DATA_ROOT}` / `${ABGS_MODELS_ROOT}` are resolved in every string.
`scenes.OPTIONS` maps stage options to dotted keys; only these keys fill options. Other keys are kept in the merged
configuration (`--print-config` shows the filled options), so a new field needs no code until a stage reads it; a
stage option is filled from a new key by adding it to `scenes.OPTIONS`.

**A stage** is a module `abundance_gs/<module>.py` with `run(args)` and an entry `Stage(module, env, gpu_mib,
description, options)` in `cli.STAGES`, where `options(parser)` adds its command-line options. For `run` to skip it
when its output is current, the stage writes `records.complete(...)` at the end and gets a record type and an inputs
function in `records.STAGE_RECORDS`; `pipeline.run` chains stages with `ensure(name, gpu, *options)`, which returns
the stage record whose output fields feed the next stage. Replacing a stage keeps its name and record fields.

**A time dimension** is not modelled; the code assumes a static scene. It would enter at these points:

- data: every stage addresses a view by its index `i` (`Pack.K(i)`, `c2w(i)`, `hsi(i)`, `rgb[i]`); a per-view field
  written into `pack.json` by `export._view` is available as `Pack.views[i][field]`;
- hit lists: `model.Hits.trace` rasterizes the frozen Gaussians for the camera of view `i`;
- per-hit feature query: `AbundanceField.encode_local_features(ids, xyz)` (local feature f_i(r));
- decoder: `AbundanceField.decoder_outputs(encoded, direction_code)` receives its inputs from `render_hits` and
  `render_view_fast`; `render_view_fused` and `csrc/fused_render.cu` reproduce the same decoder in CUDA;
- supervision: `train.Sampler` draws a view and pixels; `clusters.sample_train_spectra` pools all TRAIN views;
- evaluation: `evaluate.score_benchmark` and the scorers score each view on its own and pool the counts.

Paper symbols and code names:

| paper | code |
|---|---|
| RGB anchor; feature refinement (Stage 1) | stage `anchor` (`anchor.py`); stage `source` (`source.py`, `nht/trainer.py`) |
| harmonic features f_im; local feature f_i(r); barycentric β_im | `AbundanceField.features` [N, 48]; `encode_local_features`; `barycentric_weights` |
| γ(f), directional code Y(d_i) | `encode_local_features` (sin/cos); `directional_code`, added through `camera_adapter` |
| decoder D_φ, outputs z_i | `AbundanceField.network`; `decoder_outputs` |
| abundances a_i, intensity s_i, intersection spectrum h_i | `decode()['abundance']`, `['intensity']`, `['spectrum']` |
| endmembers Ē = [ē_1..ē_K], logits η_k | `AbundanceField.endmembers()`; `global_endmember_logits` |
| intersections of a ray; visibility weights w_i | hit lists (`model.Hits`: Gaussian `ids`, local `xyz`, `weight`) |
| rendered Ĥ(r), Â(r) | `render_hits()['hsi']`, `['abundance']` (whole views: `render_view*`) |
| descriptor q_r, brightness weight κ; pseudo-labels y_r | `clusters.appearance_descriptors`, `encoding_statistics`; `clusters.load_pseudo_labels` |
| L_rec + λ_SAM L_SAM; L_p, λ_p(t) | `train.spectral_loss` (`--w-sam`); `train.appearance_regularization`, `regularization_weight` (`--reg-floor`) |
| endmember freeze (3k updates) | `--freeze-endmembers` |
| mIoU (TRAIN-fitted many-to-one map), EM, SAM | `scorers/benchmark.py` and the other scorers; `metrics.unified_hsi_metrics` |

Third-party code and licences: `abundance_gs/third_party/THIRD_PARTY.md`.

## License

MIT License (see [LICENSE](LICENSE)). Third-party code keeps its own licences: `abundance_gs/third_party/THIRD_PARTY.md`; this includes the adapted files `abundance_gs/scene_loader.py` and `abundance_gs/nht/trainer.py`. The components derived from 3D Gaussian Splatting (Inria) are restricted to non-commercial research use.

## Citation

```bibtex
@article{lee2026swr,
  title   = {Spectral World Renderer: 3D Hyperspectral Unmixing},
  author  = {Lee, Chia-Ming and Hsiao, Yu-Jou and Chang, Ming-Ching and
             Li, Xin and Liu, Yu-Lun and Hsu, Chih-Chung},
  journal = {Preprint, under review},
  year    = {2026},
  url     = {https://ming053l.github.io/Spectral-World-Renderer/}
}
```

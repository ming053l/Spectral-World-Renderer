"""Every default of the published configuration, in one place. CLI options are generated from these."""
from dataclasses import dataclass, replace

GPU_MARGIN_GIB = 2.                      # free GPU memory left to other processes (GPU choice, hit-list placement)


@dataclass(frozen=True)
class Anchor:
    iterations: int = 30000              # 30k-step FastGS schedule
    seed: int = 0
    densification_interval: int = 500
    grad_abs_thresh: float = .0012


@dataclass(frozen=True)
class Source:
    steps: int = 5000                    # RGB feature refinement of the harmonic-texture field, geometry trained
    seed: int = 42
    feature_dim: int = 48                # 4 tetrahedron vertices x 12 harmonic dims per Gaussian


@dataclass(frozen=True)
class Clusters:
    signal_threshold: float = .03        # TRAIN pixels with smoothed spectral norm <= this are not assigned
    iterations: int = 20                 # constrained Lloyd updates after joint farthest-point seeding
    samples_per_view: int = 2304
    seed: int = 42


@dataclass(frozen=True)
class Decoder:
    """Standard decoder (paper Sec. 3.3 / 4). The `reference` preset below trains without the appearance regularization."""
    materials: int = 16                  # K; the paper's K of every scene is in configs/scenes (decoder.materials)
    steps: int = 30000
    pixels: int = 2048                   # pixels of one random TRAIN view per update
    seed: int = 42
    lr: float = 1e-3                     # cosine 1e-3 -> 1e-5 over all steps; endmembers constant 1e-3
    lr_final: float = 1e-5
    ema_decay: float = .99
    w_sam: float = .05                   # lambda_SAM: loss = L_rec + w_sam L_SAM (+ lambda_p(t) L_p)
    init: str = 'clusters'               # appearance-cluster centres ('fps': angular farthest-point, reference preset)
    reg: bool = True                     # persistent assignment regularization L_p (NLL of the pseudo-labels y_r)
    reg_weight: float = .05              # lambda_p(0); fades linearly to 0 over reg_steps ...
    reg_steps: int = 3000
    reg_floor: float = .005              # ... but never below this floor (0 = fading-only ablation)
    freeze_endmembers: int = 3000        # dictionary frozen for the first updates (300 without L_p)
    hit_cache_gib: float | str = 'auto'  # GiB of TRAIN hit lists on the GPU ('auto': free memory; rest in pinned host memory)
    gpu_memory_budget_mib: int = 10500   # raised with hit_cache_gib 'auto' when the card has room (unless given)
    working_gib: float = 3.              # GPU memory kept for training and full-view renders besides the hit lists
    margin_gib: float = GPU_MARGIN_GIB   # free memory left to other processes by hit_cache_gib 'auto'


# without the appearance regularization (real captures, K = 16; SWR-Bench reference rows, K = classes)
REFERENCE = dict(init='fps', reg=False, freeze_endmembers=300)
PRESETS = {'standard': Decoder(), 'reference': replace(Decoder(), **REFERENCE)}


@dataclass(frozen=True)
class Render:
    fast_tau: float = 1e-4               # fast renderer: hits with compositing weight < tau are skipped, MLP in fp16
    fps_repeats: int = 3                 # timing passes over the TEST views, first discarded

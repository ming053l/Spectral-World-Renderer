"""Stage `train`: the abundance decoder on the frozen source field.

Per update: one random HSI TRAIN view, `pixels` uniform random pixels, their traced hit lists decoded and composited
(model.render_hits); loss L = L_rec + lambda_SAM L_SAM + lambda_p(t) L_p, where L_rec is the L1 spectral error,
L_SAM the mean spectral angle (w_sam = lambda_SAM = .05) and L_p, the persistent assignment regularization, the NLL of
the rendered abundances against the pseudo-labels y_r, with weight lambda_p(t) = max(0.05 (1 - t/3000), floor)
(zero-based update t). The endmember dictionary is frozen for the first 3,000 updates (300 without L_p). Adam, cosine LR
1e-3 -> 1e-5 over all steps (endmembers constant 1e-3), EMA 0.99. Geometry is frozen; features, MLP, camera adapter
and endmembers are trained.

Presets: `standard` (default; the published configuration) and `reference` (no appearance regularization: angular
farthest-point initialization and a 300-step freeze; real captures). Explicit options override the preset.

Output: model_ema.pt (deployable EMA model), protocol.json, trace.json, summary.json, status.json (TEST metrics of
raw and EMA weights at the pack's resolution: unified_hsi_metrics, macro mean over views; FPS), optional curve.json
(--eval-steps) and predictions/ (--keep-cubes).
"""
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import clusters, metrics, records
from .data import Pack
from .defaults import PRESETS
from .inference import ViewRenderer, abundance_npz, cube, macro_mean, render_times, safe_name
from .model import FEATURE_DIM, GEOMETRY, AbundanceField, Hits, render_hits

_OPTIONS = ('materials', 'steps', 'pixels', 'seed', 'w_sam', 'init',
            'reg_floor', 'freeze_endmembers', 'hit_cache_gib', 'gpu_memory_budget_mib')
CURVE_KEYS = ('hsi_psnr', 'hsi_sam_deg', 'hsi_sam_p95_deg')


# ---------------------------------------------------------------- configuration and checks (CPU)

def config(args):
    """Preset defaults overridden by every explicitly given option."""
    cfg = PRESETS[args.preset]
    given = {k: getattr(args, k) for k in _OPTIONS + ('reg',) if getattr(args, k, None) is not None}
    if 'reg' in given and 'freeze_endmembers' not in given:
        given['freeze_endmembers'] = 3000 if given['reg'] else 300
    return replace(cfg, **given)


def uses_clusters(cfg):
    return cfg.reg or cfg.init == 'clusters'


def preflight(args, cfg):
    """All argument, pack and cluster-cache checks, on the CPU, before CUDA is touched.

    Returns the pack (train = HSI-supervision views), all RGB TRAIN views, validation views to score, cache audit.
    """
    if not math.isfinite(cfg.reg_floor) or cfg.reg_floor < 0:
        raise ValueError('--reg-floor must be finite and nonnegative')
    if cfg.materials <= 0 or cfg.steps <= 0 or cfg.pixels <= 0 or cfg.gpu_memory_budget_mib <= 0:
        raise ValueError('K, update count, pixels and memory budget must be positive')
    if cfg.hit_cache_gib != 'auto':
        if not math.isfinite(cfg.hit_cache_gib) or cfg.hit_cache_gib < 0:
            raise ValueError("--hit-cache-gib must be 'auto' or finite and nonnegative")
        need = math.ceil((cfg.hit_cache_gib + cfg.working_gib) * 1024)
        if need > cfg.gpu_memory_budget_mib:
            raise ValueError(f'--hit-cache-gib {cfg.hit_cache_gib:g} does not fit --gpu-memory-budget-mib '
                             f'{cfg.gpu_memory_budget_mib} (hit lists + {cfg.working_gib:g} GiB working set = {need} MiB): '
                             f"raise the budget to at least {need}, lower --hit-cache-gib, or use --hit-cache-gib auto")
    geometry = args.source if args.source is not None else args.anchor_ply
    if geometry is None:
        raise ValueError('Give exactly one of --source or --anchor-ply')
    if not geometry.is_file() or not os.access(geometry, os.R_OK):
        raise ValueError('Geometry source is not a readable file: ' + str(geometry))
    if (args.output / 'status.json').exists() or (args.output / 'model_ema.pt').exists():
        raise FileExistsError('Preserve the existing trained output: ' + str(args.output))
    pack = Pack(args.pack)
    rgb_train, held_out = pack.limit_hsi_training(args.train_max_views, args.val_every_n_train)
    audit = None
    if not uses_clusters(cfg):
        args.clusters = None                           # neither initialization nor L_p reads a cache
    else:
        cache = args.clusters if args.clusters is not None else args.output / 'appearance_clusters'
        args.clusters, _, audit = clusters.ensure_cache(pack, cache, cfg.materials)
    return pack, rgb_train, held_out if args.score_val else [], audit


# ---------------------------------------------------------------- inputs

def load_geometry(args, cfg, pack, rgb_train, device):
    """Frozen geometry + initial features: the source checkpoint, or (--anchor-ply) the dilated anchor and random features."""
    if args.source is not None:
        source = torch.load(args.source, map_location='cpu', weights_only=True)
        splats = {k: v.float().to(device) for k, v in source['splats'].items()}
        return splats, {'source': str(args.source.resolve()), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest()}
    from .source import dilated_anchor
    anchor, _, _ = dilated_anchor(args.anchor_ply, pack, rgb_train)
    generator = torch.Generator().manual_seed(cfg.seed)
    features = torch.rand(len(anchor['means']), FEATURE_DIM, generator=generator) * math.pi - math.pi / 2     # NHT feature init
    splats = {k: anchor[k].float().to(device) for k in GEOMETRY}
    splats['features'] = features.to(device)
    return splats, {'source': None, 'anchor_ply': str(args.anchor_ply.resolve()), 'no_rgb_source_stage': True}


class Targets:
    """HSI targets [H*W, B] of the used views on the CPU (float16 above 48 GiB) and the TRAIN mean spectrum."""

    def __init__(self, pack, views):
        self.pack = pack
        self.gib_float32 = len(pack.views) * pack.pixels * pack.bands * 4 / 2 ** 30
        self.half = self.gib_float32 > 48
        self.flat = {}
        mean_sum = torch.zeros(pack.bands, dtype=torch.float64)
        for i, cube in pack.hsi_iter(views):
            flat = cube.permute(1, 2, 0).reshape(-1, pack.bands).contiguous()
            if i in pack.train:
                mean_sum += flat.double().sum(0)
            self.flat[i] = flat.half() if self.half else flat
        self.mean = (mean_sum / (len(pack.train) * pack.pixels)).float()

    def pixels(self, i, pix, device):
        return self.flat[i][pix].float().to(device)

    def cube(self, i, device):
        return cube(self.flat[i].float().to(device), self.pack)


def initial_dictionary(cfg, cache, pack):
    """Initial endmembers [K, B], pseudo-labels y_r per TRAIN view and the cache's preparation record."""
    labels, preparation = {}, None
    if uses_clusters(cfg):
        initial, labels, preparation = clusters.load_pseudo_labels(cache, pack, cfg.materials)
    if cfg.init == 'fps':
        initial = clusters.farthest_point_endmembers(pack, cfg.materials)
    return initial, labels, preparation


def build_optimizer(model, cfg):
    groups = [{'params': [model.features], 'lr': cfg.lr, 'name': 'features'}]
    groups += [{'params': list(module.parameters()), 'lr': cfg.lr, 'name': name} for name, module in model.trainable_modules()]
    groups.append({'params': [model.global_endmember_logits], 'lr': cfg.lr, 'name': 'endmembers'})
    assert {id(p) for p in model.parameters()} == {id(p) for g in groups for p in g['params']}
    return torch.optim.Adam(groups)


# ---------------------------------------------------------------- schedules, EMA

def regularization_weight(step, cfg):
    """lambda_p(t): weight of the persistent assignment regularization L_p at a zero-based update t."""
    if not cfg.reg:
        return 0.
    fading = cfg.reg_weight * (1 - step / cfg.reg_steps) if step < cfg.reg_steps else 0.
    return max(fading, cfg.reg_floor)


def cosine_lr(step, total, initial=1e-3, final=1e-5):
    t = min(max(step, 0), total - 1) / max(total - 1, 1)
    return final + (initial - final) * .5 * (1 + math.cos(math.pi * t))


def create_ema(model):
    return {name: {k: v.detach().clone() for k, v in module.state_dict().items()} for name, module in model.trainable_modules()}


class EMA:
    """Exponential moving average of the MLP and camera adapter (their state_dict tensors, all floating point)."""

    def __init__(self, model, decay):
        self.decay, self.state = decay, create_ema(model)
        pairs = [(self.state[name][k], v) for name, module in model.trainable_modules() for k, v in module.state_dict().items()]
        assert all(v.is_floating_point() for _, v in pairs)
        self.averages, self.values = [a for a, _ in pairs], [v for _, v in pairs]

    @torch.no_grad()
    def update(self):
        torch._foreach_mul_(self.averages, self.decay)
        torch._foreach_add_(self.averages, self.values, alpha=1 - self.decay)


@contextmanager
def use_ema(model, ema):
    """EMA covers the MLP and camera adapter; features and endmembers stay at their raw values."""
    original = create_ema(model)
    for name, module in model.trainable_modules():
        module.load_state_dict(ema[name], strict=True)
    try:
        yield
    finally:
        for name, module in model.trainable_modules():
            module.load_state_dict(original[name], strict=True)


# ---------------------------------------------------------------- one update

class Sampler:
    """One random HSI TRAIN view and uniform pixels per update; SHA256 over the whole sampled sequence."""

    def __init__(self, pack, cfg):
        self.views, self.pixels, self.count = pack.train, pack.pixels, cfg.pixels
        self.rng = torch.Generator().manual_seed(cfg.seed)
        self.digest = hashlib.sha256()

    def __call__(self, step):
        view = self.views[int(torch.randint(len(self.views), (1,), generator=self.rng))]
        pix = torch.randint(self.pixels, (self.count,), generator=self.rng)
        self.digest.update(np.asarray([step, view], dtype='<i8').tobytes())
        self.digest.update(pix.numpy().astype('<i8', copy=False).tobytes())
        return view, pix


def spectral_loss(pred, target, w_sam):
    """L_rec + lambda_SAM L_SAM: L1 error plus w_sam times the mean spectral angle (rad) over pixels with a nonzero
    target; returns (loss, L_rec, L_SAM)."""
    valid = target.norm(dim=-1) > 1e-6
    l1 = F.l1_loss(pred, target)
    sam = (F.cosine_similarity(pred[valid], target[valid], dim=-1).clamp(-1 + 1e-6, 1 - 1e-6).acos().mean()
           if valid.any() else pred.sum() * 0.)
    return l1 + w_sam * sam, l1, sam


def appearance_regularization(rendered, labels):
    """L_p: NLL of the rendered abundances against the pseudo-labels y_r of covered, labelled pixels (None if none)."""
    eligible = (labels >= 0) & (rendered['alpha'].squeeze(-1) > 1e-6)
    if not eligible.any():
        return None
    return F.nll_loss(rendered['abundance'][eligible].clamp_min(1e-8).log(), labels[eligible])


# ---------------------------------------------------------------- run

def gpu_memory(cfg, budget_given):
    """(memory budget MiB, GPU hit-list GiB). 'auto': free memory minus the margin, less the working set; the budget is
    raised to the free memory minus the margin unless it was given. Only where hit lists live changes, not the hits."""
    if cfg.hit_cache_gib != 'auto':
        return cfg.gpu_memory_budget_mib, cfg.hit_cache_gib
    free, total = torch.cuda.mem_get_info()
    available = max(0., free / 2 ** 20 - cfg.margin_gib * 1024)
    budget = cfg.gpu_memory_budget_mib if budget_given else max(cfg.gpu_memory_budget_mib, int(min(available, total / 2 ** 20)))
    return budget, max(0., (min(budget, available) - cfg.working_gib * 1024) / 1024)


def run(args):
    cfg = config(args)
    pack, rgb_train, val_views, cache_audit = preflight(args, cfg)
    record_inputs, record_config = records.expected('train', args)
    budget_mib, hit_cache_gib = gpu_memory(cfg, args.gpu_memory_budget_mib is not None)
    print(f'GPU memory budget {budget_mib} MiB, hit lists on the GPU up to {hit_cache_gib:.2f} GiB'
          + (' (auto)' if cfg.hit_cache_gib == 'auto' else ''), flush=True)
    torch.cuda.set_per_process_memory_fraction(min(1., budget_mib * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory))
    eval_steps = sorted({int(v) for v in args.eval_steps.split(',') if v})
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    splats, source_record = load_geometry(args, cfg, pack, rgb_train, device)

    load_start = time.perf_counter()                   # targets, TRAIN mean and initial dictionary: HSI TRAIN views only
    targets = Targets(pack, pack.train + pack.test + val_views)
    initial, labels, preparation = initial_dictionary(cfg, args.clusters, pack)
    labels = {i: y.to(device) for i, y in labels.items()}                 # pseudo-labels, gathered on the GPU each update
    centers = {i: pack.center(i, device) for i in pack.train + pack.test + val_views}
    data_seconds = time.perf_counter() - load_start

    model = AbundanceField(splats['features'], initial.to(device), targets.mean.to(device)).to(device)
    model.set_primitive_means(splats['means'])
    optimizer = build_optimizer(model, cfg)
    sampler = Sampler(pack, cfg)
    ema = EMA(model, cfg.ema_decay)
    hits = Hits(splats, pack, hit_cache_gib)
    renderer = ViewRenderer(model, hits, pack)
    renderer.centers = centers
    protocol = protocol_record(args, cfg, pack, rgb_train, splats, source_record, targets, preparation, cache_audit)
    write_json(args.output / 'protocol.json', protocol, indent=2)

    def score(views):
        rows = [metrics.unified_hsi_metrics(cube(renderer(i)[0], pack), targets.cube(i, device), 1e-6) for i in views]
        return macro_mean(rows, CURVE_KEYS)

    trace, curve = [], []
    eval_seconds = 0.
    nonfinite = torch.zeros((), dtype=torch.bool, device=device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    hits.preload(pack.train)                           # every HSI TRAIN view traced once (counted in the training time)
    for step in range(cfg.steps):
        for group in optimizer.param_groups:
            if group['name'] != 'endmembers':
                group['lr'] = cosine_lr(step, cfg.steps, cfg.lr, cfg.lr_final)
        model.global_endmember_logits.requires_grad_(step >= cfg.freeze_endmembers)
        view, pix = sampler(step)
        pixels = pix.to(device)
        hit_lists, ids = hits.select(view, pixels)
        optimizer.zero_grad(set_to_none=True)
        rendered = render_hits(model, hit_lists, ids, len(pix), centers[view])
        loss, l1, sam = spectral_loss(rendered['hsi'], targets.pixels(view, pix, device), cfg.w_sam)
        weight = regularization_weight(step, cfg)
        nll = appearance_regularization(rendered, labels[view][pixels]) if weight > 0 else None
        if nll is not None:
            loss = loss + weight * nll
        nonfinite |= ~torch.isfinite(loss.detach())    # checked at the trace steps: no host sync per update
        loss.backward()
        optimizer.step()
        ema.update()
        if step % 100 == 0 or step == cfg.steps - 1:
            if nonfinite:
                raise FloatingPointError(f'Nonfinite loss at or before update {step}')
            trace.append({'step': step + 1, 'loss': float(loss.detach()), 'l1': float(l1.detach()), 'sam_rad': float(sam.detach()),
                          'reg_weight': weight, 'reg_nll': None if nll is None else float(nll.detach()),
                          'sample_sequence_sha256': sampler.digest.hexdigest()})
            if step % 2000 == 0:
                print(trace[-1], f'traces={hits.traces} gpu_views={len(hits.cache)} host_views={len(hits.host)}', flush=True)
            if args.live_every:
                live_row(args.output, trace[-1], cfg.steps)
        if args.live_every and (step + 1) % args.live_every == 0 and step + 1 < cfg.steps:
            with torch.no_grad(), use_ema(model, ema.state):
                live_model(args.output, deployable_state(model, splats, initial, targets.mean), step + 1)
        if step + 1 in eval_steps:                     # TEST curve with EMA weights; the clock is stopped
            torch.cuda.synchronize()
            pause = time.perf_counter()
            with torch.no_grad(), use_ema(model, ema.state):
                row = {'step': step + 1, **score(pack.test)}
            row['training_seconds'] = pause - begin - eval_seconds
            curve.append(row)
            print('CURVE', row, flush=True)
            torch.cuda.synchronize()
            eval_seconds += time.perf_counter() - pause
    torch.cuda.synchronize()
    training = dict(seconds=time.perf_counter() - begin - eval_seconds, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                    traces=hits.traces, trace_seconds=hits.trace_seconds)
    protocol['training_sample_sequence'].update(completed_updates=cfg.steps, sha256=sampler.digest.hexdigest())
    write_json(args.output / 'protocol.json', protocol, indent=2)
    if curve:
        write_json(args.output / 'curve.json', curve, indent=1)
    (args.output / 'trace.json').write_text(json.dumps(trace) + '\n')

    # TEST evaluation at the pack resolution: raw and EMA weights, canonical metric, macro mean over views
    predictions = args.output / 'predictions' if args.keep_cubes else None
    with torch.no_grad():
        summary = {'raw': evaluate(renderer, targets, device)}
        with use_ema(model, ema.state):
            summary['ema'] = evaluate(renderer, targets, device, dominant=True, predictions=predictions)
            torch.save(deployable_state(model, splats, initial, targets.mean), args.output / 'model_ema.pt')
            times = render_times(renderer, pack.test)      # full pipeline, EMA weights, FP32
            fps = len(times) / sum(times)
        metrics_val = None
        if val_views:
            with use_ema(model, ema.state):
                metrics_val = score(val_views)
            metrics_val['views'] = [pack.names[i] for i in val_views]
    status = status_record(args, cfg, pack, rgb_train, model, splats, hits, summary, training, data_seconds, fps, curve,
                           metrics_val, trace)
    write_json(args.output / 'summary.json', summary, indent=1)
    write_json(args.output / 'status.json', status, indent=2, atomic=True)
    records.complete(args.output, 'train', record_inputs, record_config, model=str(args.output / 'model_ema.pt'),
                     metrics_ema=status['metrics_ema'])
    print(json.dumps({k: status[k] for k in ('metrics_ema', 'training_seconds', 'render_fps_full_pipeline_fp32')}, indent=1))


# ---------------------------------------------------------------- evaluation and records

def evaluate(renderer, targets, device, dominant=False, predictions=None):
    """Per-TEST-view canonical metrics (+ coverage; EMA: fraction of covered pixels per dominant endmember, optional
    prediction files) and their mean."""
    pack, model = renderer.pack, renderer.model
    rows = []
    for i in pack.test:
        name = pack.names[i]
        hsi, abundance, alpha = renderer(i)
        prediction = cube(hsi, pack)
        row = metrics.unified_hsi_metrics(prediction, targets.cube(i, device), 1e-6)
        covered = alpha.squeeze(-1) > 1e-6
        row.update(view=name, mean_alpha=float(alpha.mean()), uncovered_fraction=float((~covered).float().mean()))
        if dominant:
            dominant_endmember = abundance.argmax(-1)
            row['dominant_slot_fraction'] = torch.bincount(dominant_endmember[covered], minlength=model.num_materials).float().div(
                max(int(covered.sum()), 1)).tolist()
        if predictions is not None:
            predictions.mkdir(exist_ok=True)
            np.savez(predictions / f'{safe_name(name)}.npz', prediction=prediction.clamp(0, 1).float().cpu().numpy())
            np.savez_compressed(predictions / f'{safe_name(name)}_abundance.npz', **abundance_npz(abundance, alpha, pack, clamp=False))
        rows.append(row)
    return {'views': rows, 'mean': macro_mean(rows)}


def live_row(output, row, steps):
    """--live-every: the loss row of a trace step, appended to OUTPUT/live/trace.jsonl (read by `abgs viewer`)."""
    (output / 'live').mkdir(exist_ok=True)
    with open(output / 'live' / 'trace.jsonl', 'a') as f:
        f.write(json.dumps({**row, 'steps': steps}) + '\n')


def live_model(output, state, step):
    """--live-every: the current EMA model as OUTPUT/live/model_ema.pt (replaced atomically), a preview for the viewer."""
    (output / 'live').mkdir(exist_ok=True)
    temporary = output / 'live' / 'model_ema.pt.tmp'
    torch.save({**state, 'live_step': step}, temporary)
    temporary.replace(output / 'live' / 'model_ema.pt')


def deployable_state(model, splats, initial, mean):
    return {**{k: splats[k].cpu() for k in GEOMETRY},
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'endmembers': model.endmembers().cpu(), 'initial_endmembers': initial, 'mean_spectrum': mean}


def protocol_record(args, cfg, pack, rgb_train, splats, source_record, targets, preparation, cache_audit):
    cache = args.clusters
    return {
        'scene_id': pack.meta['scene_id'], 'seed': cfg.seed, 'pixels': cfg.pixels, 'steps': cfg.steps,
        'materials': cfg.materials, 'bands': pack.bands, 'resolution_wh': [pack.width, pack.height],
        'hsi_train_views': len(pack.train), 'rgb_train_views': len(rgb_train), 'test_views': len(pack.test),
        'val_every_n_train': args.val_every_n_train, 'train_max_views': args.train_max_views,
        'hsi_train_view_names': [pack.names[i] for i in pack.train], **source_record,
        'primitives': len(splats['means']),
        'targets_dtype': 'float16' if targets.half else 'float32', 'target_gib_float32': targets.gib_float32,
        'losses': {'l1': 1., 'sam_rad': cfg.w_sam}, 'hit_cache_gib': cfg.hit_cache_gib,
        'config': 'material head, large capacity, pre-blend decode, degree-2 camera SH adapter, EMA 0.99',
        'preset': args.preset, 'decoder_config': asdict(cfg), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'trainer': 'abundance_gs/train.py', 'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'appearance_regularization': {
            'enabled': cfg.reg, 'initialization': cfg.init, 'cluster_cache': None if cache is None else str(cache),
            'preparation_sha256': None if cache is None else clusters.sha(cache / 'preparation.json'),
            'clustering_algorithm': None if preparation is None else preparation.get('algorithm'),
            'weight': f'max({cfg.reg_weight}*(1-step/{cfg.reg_steps}) if step<{cfg.reg_steps} else 0, {cfg.reg_floor})',
            'step_index': 'zero-based optimizer update', 'endmembers_frozen_first_steps': cfg.freeze_endmembers,
            'sampling': 'uniform pixels', 'cache_audit': cache_audit,
            'cluster_labels': 'TRAIN-only appearance-cluster labels; observed appearance, not material GT',
            'material_GT_or_TEST_used_for_initialization_or_regularization': False},
        'training_sample_sequence': {
            'algorithm': 'SHA256',
            'definition': 'per update: little-endian int64 [zero-based step, Pack TRAIN view index], then actual sampled pixel indices in order',
            'sampling_consumes_no_extra_RNG': True, 'expected_updates': cfg.steps, 'expected_pixels_per_update': cfg.pixels,
            'completed_updates': 0, 'sha256': None}}


def status_record(args, cfg, pack, rgb_train, model, splats, hits, summary, training, data_seconds, fps, curve, metrics_val, trace):
    geometry_bytes = sum(splats[k].numel() * 4 for k in GEOMETRY)
    network_bytes = sum(p.numel() * 4 for name, module in model.trainable_modules() for p in module.parameters())
    return {'status': 'complete', 'steps': cfg.steps, 'seed': cfg.seed, 'scene_id': pack.meta['scene_id'],
            'metrics_ema': summary['ema']['mean'], 'metrics_raw': summary['raw']['mean'],
            'training_seconds': training['seconds'], 'data_loading_seconds': data_seconds,
            'trace_seconds_in_training': training['trace_seconds'], 'traces_in_training': training['traces'],
            'cached_views': len(hits.cache), 'host_views': len(hits.host), 'hit_cache_gib_used': hits.used / 2 ** 30,
            'hit_cache_gib_limit': hits.budget / 2 ** 30,
            'hits_per_pixel_mean': float(np.mean(list(hits.hits_per_pixel.values()))),
            'training_peak_allocated_gib': training['peak_gib'], 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
            'render_fps_full_pipeline_fp32': fps, 'primitives': len(splats['means']),
            'model_ema_bytes': (args.output / 'model_ema.pt').stat().st_size,
            'size_breakdown_bytes': {'geometry_fp32': geometry_bytes, 'features_fp32': model.features.numel() * 4,
                                     'features_fp16': model.features.numel() * 2, 'network_and_camera_adapter_fp32': network_bytes,
                                     'endmembers_fp32': model.global_endmember_logits.numel() * 4},
            'parameter_counts': model.parameter_counts(),
            'hsi_train_views': len(pack.train), 'rgb_train_views': len(rgb_train), 'curve': curve,
            'metrics_val_ema': metrics_val, 'materials': cfg.materials,
            'final_train_loss_last10': {k: float(np.mean([r[k] for r in trace[-10:]])) for k in ('l1', 'sam_rad', 'loss')},
            'selection': 'final step; EMA reported, raw also given; no checkpoint or view selection'}


def write_json(path, value, indent=1, atomic=False):
    target = path.with_suffix(path.suffix + '.tmp') if atomic else path
    target.write_text(json.dumps(value, indent=indent) + '\n')
    if atomic:
        target.replace(path)

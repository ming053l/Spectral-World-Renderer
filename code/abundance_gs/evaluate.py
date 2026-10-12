"""Stage `eval`: render a trained decoder on a (typically native-resolution) pack and score it.

Benchmark layout (--clean-dataset given; SWR-Bench scenes): every TEST and TRAIN view of the pack is rendered at its
own resolution and intrinsics (e.g. trained at 480x270, scored at 960x540) and written to <output>/
  spectral_metrics.json  TEST: canonical unified_hsi_metrics vs the clean and the noisy cube (both divided by the
                         dataset's TRAIN-only scale, clamp [0,1]), GT-foreground and unclipped variants, noise floor
                         (noisy vs clean); TRAIN: the same metrics (train/test gap)
  predictions/<view>_abundance.npz  float16 abundance [K,H,W] + alpha [H,W] (material scorer input)
  endmembers_radiance.npy           K x B learned endmembers x scale (radiance units)
With --scorer NAME the benchmark material scorer abundance_gs/scorers/NAME.py then runs on these files (command line
--scorer-args with {output} {predictions} {endmembers} {dataset} {clean_dataset} {report} filled in) and writes
benchmark_material_evaluation.json (TRAIN-calibrated many-to-one endmember->material mapping, fixed for TEST).
--retain-views then deletes every abundance map except those views.

Other packs (no --clean-dataset): TEST views are scored against the pack's own HSI with the canonical metric.
--fast uses the fp16 / weight<1e-4-skipping renderer (identical scores to < 1e-3).
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import records
from .metrics import unified_hsi_metrics
from .scorers import SCORERS
from .data import prefetch
from .inference import BackgroundWriter, abundance_npz, cube, macro_mean, open_model, timed

SCORER_ARGS = '--dataset {dataset} --predictions {predictions} --endmembers {endmembers} --split test --output {report}'
UNIFIED = ('hsi_psnr', 'hsi_sam_deg', 'hsi_sam_p95_deg', 'hsi_foreground_psnr')
MASKED = ('psnr', 'sam_deg', 'sam_p95_deg')
UNCLIPPED = ('psnr_all_pixels', 'psnr', 'sam_deg', 'sam_p95_deg')


def sam_degrees(p, g):
    """Per-pixel spectral angle (deg) of [B, N] columns, over pixels whose reference norm > 1e-6."""
    ok = g.norm(dim=0) > 1e-6
    cos = (p[:, ok] * g[:, ok]).sum(0) / (p[:, ok].norm(dim=0).clamp_min(1e-8) * g[:, ok].norm(dim=0).clamp_min(1e-8))
    return torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)))


def masked_stats(pred, gt, mask):
    """pred / gt: [B, N] clamped to [0,1]; mask: [N] bool. PSNR over masked pixels, SAM over masked pixels with gt norm > 1e-6."""
    p, g = pred[:, mask], gt[:, mask]
    mse = float((p - g).square().mean())
    sam = sam_degrees(p, g)
    return {'psnr': -10 * np.log10(max(mse, 1e-10)), 'sam_deg': float(sam.mean()), 'sam_p95_deg': float(torch.quantile(sam, .95)),
            'pixels': int(mask.sum())}


def unclipped(pred, gt, mask):
    """pred / gt: [B, H, W] scaled, not clamped. PSNR (peak 1) over all pixels and over the mask; SAM over the mask."""
    B = pred.shape[0]
    p, g = pred.reshape(B, -1).float(), gt.reshape(B, -1).float()
    mse_all = float((p - g).square().mean())
    out = masked_stats(p, g, mask)
    out['psnr_all_pixels'] = -10 * np.log10(max(mse_all, 1e-12))
    out['reference_max'] = float(g.max())
    return out


class ClassSpectral:
    """Pooled TEST spectral metrics per GT first-interface class / opaque group (--class-spectral, Karst)."""

    def __init__(self, dataset):
        proto = json.loads((dataset / 'material_class_protocol.json').read_text())
        self.names = list(proto['classes'])
        opaque = [self.names.index(c) for c in proto.get('opaque_reference_endmember_classes', [])]
        self.groups = {n: [i] for i, n in enumerate(self.names)}
        if opaque:
            self.groups['opaque_all'] = opaque
            self.groups['non_opaque_reference'] = [i for i in range(len(self.names)) if i not in opaque]
        self.acc = {}

    def _add(self, key, p, g, lab):
        for grp, ids in self.groups.items():
            m = torch.zeros_like(lab, dtype=torch.bool)
            for i in ids:
                m |= lab == i
            if not bool(m.any()):
                continue
            a = self.acc.setdefault((key, grp), {'sse': 0.0, 'n': 0, 'px': 0, 'sam': []})
            pp, gg = p[:, m], g[:, m]
            a['sse'] += float((pp - gg).double().square().sum())
            a['n'] += int(pp.numel())
            a['px'] += int(m.sum())
            a['sam'].append(sam_degrees(pp, gg).float().cpu())

    def add(self, gt_npz, pred_clamped, refs, noisy):
        with np.load(gt_npz) as z:
            lab = torch.from_numpy(z['material'].astype(np.int64)).to(pred_clamped.device).reshape(-1)
        B = pred_clamped.shape[0]
        for key, r in refs.items():
            self._add(key, pred_clamped, r.clamp(0, 1).reshape(B, -1), lab)
        self._add('noisy_observation_vs_clean', noisy.clamp(0, 1).reshape(B, -1), refs['clean'].clamp(0, 1).reshape(B, -1), lab)

    def summary(self):
        out = {'class_names': self.names, 'groups': {g: [self.names[i] for i in ids] for g, ids in self.groups.items()},
               'definition': 'pooled over all TEST pixels with GT material in the group; prediction and reference clamped to [0,1] '
                             '(x / TRAIN-only scale), PSNR peak 1 from the pooled MSE; SAM over pixels with reference norm > 1e-6'}
        for (key, grp), a in self.acc.items():
            sam = torch.cat(a['sam']) if a['sam'] else torch.zeros(0)
            out.setdefault(key, {})[grp] = {
                'psnr': -10 * np.log10(max(a['sse'] / max(a['n'], 1), 1e-12)), 'pixels': a['px'],
                'sam_deg': float(sam.mean()) if sam.numel() else None,
                'sam_p95_deg': float(np.percentile(sam.numpy(), 95)) if sam.numel() else None}
        return out


class BenchmarkReferences:
    """Clean / noisy reference cubes and GT foreground masks of one benchmark view, in the pack's scaled units."""

    def __init__(self, pack, clean_dataset, clean_hsi_subdir, foreground, device):
        self.noisy_dataset, self.clean_dataset = Path(pack.meta['source_path']), clean_dataset
        self.clean_subdir, self.noisy_subdir = clean_hsi_subdir, pack.meta.get('hsi_subdir', 'hsi')
        self.scale = float(pack.meta['hsi_normalization_scale'])
        self.multiplier = pack.meta.get('hsi_scale_multiplier')
        self.foreground_kind, self.device = foreground, device
        self.light_class = light_source_class(self.noisy_dataset)

    def _read(self, path):
        x = np.load(path)
        return x / self.scale if self.multiplier is None else x * np.float32(self.multiplier)

    def to_device(self, x):
        """[H, W, B] scaled array -> [B, H, W] float32 on the device."""
        return torch.from_numpy(x).float().to(self.device).permute(2, 0, 1).contiguous()

    def read(self, name):
        """Scaled clean and noisy cubes of a view as host arrays (thread-safe; see `prefetch`)."""
        return (self._read(self.clean_dataset / self.clean_subdir / f'{name}.npy'),
                self._read(self.noisy_dataset / self.noisy_subdir / f'{name}.npy'))

    def gt(self, name):
        return self.noisy_dataset / 'gt' / f'{name}.npz'

    def foreground(self, name):
        with np.load(self.gt(name)) as z:
            if self.foreground_kind == 'surface':
                mask = z['surface_evaluation_mask'].astype(bool)
            elif 'coverage' in z.files:
                mask = z['coverage'] > 0
            else:
                mask = z['material_id'] < self.light_class
        return torch.from_numpy(mask).to(self.device).reshape(-1)

    @property
    def rule(self):
        if self.foreground_kind == 'surface':
            return 'released surface_evaluation_mask (coverage >= 0.5)'
        return f'coverage > 0, else material_id < {self.light_class}'


def light_source_class(dataset):
    """GT foreground without 'coverage' = material_id < light-source class (Office protocol: 8; Study layout: 5)."""
    protocol = dataset / 'scene_protocol.json'
    if protocol.exists():
        return int(json.loads(protocol.read_text()).get('light_source_class', 5))
    return 5


def run(args):
    if args.clean_dataset is None and (args.scorer is not None or args.class_spectral):
        raise ValueError('--scorer / --class-spectral need the benchmark layout (--clean-dataset)')
    record_inputs, record_config = records.expected('eval', args)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_budget_mib * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory)
    renderer, model, _, pack, hits = open_model(args.model, args.pack, 'fused' if args.fused else args.fast)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.clean_dataset is None:
        result = score_pack(pack, renderer)
    else:
        result = score_benchmark(args, pack, model, renderer)
    result.update(model=str(args.model), renderer=renderer.description,
                  hits_per_pixel_mean=float(np.mean(list(hits.hits_per_pixel.values()))), gpu_name=torch.cuda.get_device_name())
    (args.output / 'spectral_metrics.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['summary'], indent=1))
    if args.scorer is not None:
        run_scorer(args, pack)
        if args.retain_views:
            prune_predictions(args.output / 'predictions', args.retain_views)
    scores = args.output / 'benchmark_material_evaluation.json'
    records.complete(args.output, 'eval', record_inputs, record_config,
                     spectral_metrics=str(args.output / 'spectral_metrics.json'),
                     material_scores=str(scores) if scores.exists() else '')


def scorer_command(args, pack):
    """`python -m abundance_gs.scorers.<scorer>` with the template filled in (scorers/: the benchmark's scorers)."""
    if args.scorer not in SCORERS:
        raise ValueError(f'Unknown scorer {args.scorer!r}; one of {SCORERS}')
    template = args.scorer_args if args.scorer_args is not None else SCORER_ARGS
    template = shlex.split(template) if isinstance(template, str) else list(template)
    fields = dict(output=args.output, predictions=args.output / 'predictions', endmembers=args.output / 'endmembers_radiance.npy',
                  dataset=args.dataset or Path(pack.meta['source_path']), clean_dataset=args.clean_dataset,
                  report=args.output / 'benchmark_material_evaluation.json')
    return [sys.executable, '-m', f'abundance_gs.scorers.{args.scorer}', *(part.format(**fields) for part in template)]


def run_scorer(args, pack):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run(scorer_command(args, pack), check=True, env=environment)
    print('material scores:', args.output / 'benchmark_material_evaluation.json')


def prune_predictions(directory, keep):
    """Delete abundance maps of every view not in `keep` (after the scorer has read them)."""
    keep = set(keep.split(',') if isinstance(keep, str) else keep)
    removed = [p for p in directory.glob('*_abundance.npz') if p.name[:-len('_abundance.npz')] not in keep]
    for p in removed:
        p.unlink()
    print(f'removed {len(removed)} abundance maps; kept {sorted(keep)}')


@torch.no_grad()
def score_pack(pack, renderer):
    rows, seconds = [], []
    for i in pack.test:
        (hsi, _, _), elapsed = timed(renderer, i)
        seconds.append(elapsed)
        rows.append({'view': pack.names[i], **unified_hsi_metrics(cube(hsi, pack), pack.hsi(i).to(renderer.device), 1e-6)})
    return {'resolution_wh': [pack.width, pack.height], 'views': rows,
            'summary': {'unified_macro_mean': macro_mean(rows)},
            'render_seconds_per_view_mean': float(np.mean(seconds)),
            'metric_notes': 'unified_hsi_metrics (metrics.py) vs the pack HSI, macro mean over TEST views'}


def reference_metrics(prediction, reference, foreground):
    """Every metric of one prediction cube against one reference cube (unified, GT foreground, unclipped)."""
    B = prediction.shape[0]
    return {'unified': unified_hsi_metrics(prediction, reference, 1e-6),
            'gt_foreground': masked_stats(prediction.clamp(0, 1).reshape(B, -1), reference.clamp(0, 1).reshape(B, -1), foreground),
            'reference_values_above_1': float((reference > 1).float().mean()),
            'reference_values_below_0': float((reference < 0).float().mean())}


def test_row(name, prediction, alpha, clean, noisy, foreground):
    row = {'view': name, 'mean_alpha': float(alpha.mean()), 'uncovered_fraction': float((alpha <= 1e-6).float().mean()),
           'gt_foreground_fraction': float(foreground.float().mean())}
    for key, reference in (('clean', clean), ('noisy', noisy)):
        row[key] = reference_metrics(prediction, reference, foreground)
        row[key + '_unclipped'] = unclipped(prediction, reference, foreground)
    # noise floor: the stored noisy observation scored against the clean cube with the same metric
    B = prediction.shape[0]
    row['noisy_vs_clean_unified'] = unified_hsi_metrics(noisy, clean, 1e-6)
    row['noisy_vs_clean_gt_foreground'] = masked_stats(noisy.clamp(0, 1).reshape(B, -1), clean.clamp(0, 1).reshape(B, -1), foreground)
    row['noisy_vs_clean_unclipped'] = unclipped(noisy, clean, foreground)
    return row


def train_row(name, prediction, clean, noisy, foreground):
    return {'view': name, 'clean': unified_hsi_metrics(prediction, clean, 1e-6),
            'noisy': unified_hsi_metrics(prediction, noisy, 1e-6),
            'clean_unclipped': unclipped(prediction, clean, foreground), 'noisy_unclipped': unclipped(prediction, noisy, foreground)}


def summarize(rows, train_rows):
    def mean(get, rs):
        return float(np.mean([get(r) for r in rs]))
    summary = {}
    for r in ('clean', 'noisy'):
        summary[r] = {'unified_macro_mean': {k: mean(lambda x: x[r]['unified'][k], rows) for k in UNIFIED},
                      'gt_foreground_macro_mean': {k: mean(lambda x: x[r]['gt_foreground'][k], rows) for k in MASKED}}
    summary['noisy_observation_vs_clean'] = {
        'unified_macro_mean': {k: mean(lambda r: r['noisy_vs_clean_unified'][k], rows) for k in UNIFIED[:3]},
        'gt_foreground_macro_mean': {k: mean(lambda r: r['noisy_vs_clean_gt_foreground'][k], rows) for k in MASKED},
        'unclipped_macro_mean': {k: mean(lambda r: r['noisy_vs_clean_unclipped'][k], rows) for k in UNCLIPPED}}
    for r in ('clean', 'noisy'):
        summary[r]['unclipped_macro_mean'] = {k: mean(lambda x: x[r + '_unclipped'][k], rows) for k in UNCLIPPED}
    summary['train'] = {r: {'unified_macro_mean': {k: mean(lambda x: x[r][k], train_rows) for k in UNIFIED},
                            'unclipped_macro_mean': {k: mean(lambda x: x[r + '_unclipped'][k], train_rows) for k in UNCLIPPED}}
                        for r in ('clean', 'noisy')}
    return summary


@torch.no_grad()
def score_benchmark(args, pack, model, renderer):
    refs = BenchmarkReferences(pack, args.clean_dataset, args.clean_hsi_subdir, args.foreground, renderer.device)
    predictions = args.output / 'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'endmembers_radiance.npy', (model.endmembers().float().cpu().numpy() * refs.scale).astype(np.float64))
    classes = ClassSpectral(refs.noisy_dataset) if args.class_spectral else None
    rows, train_rows, render_seconds, t0 = [], [], [], time.time()
    # reference cubes are read ahead and abundance maps compressed in threads, overlapping the GPU work
    with BackgroundWriter() as writer:
        for i, (clean, noisy) in prefetch(pack.test + pack.train, lambda i: refs.read(pack.names[i]), workers=2):
            name = pack.names[i]
            (hsi, abundance, alpha), elapsed = timed(renderer, i)
            render_seconds.append(elapsed)
            writer.submit(np.savez_compressed, predictions / f'{name}_abundance.npz', **abundance_npz(abundance, alpha, pack))
            prediction = cube(hsi, pack)
            clean, noisy, foreground = refs.to_device(clean), refs.to_device(noisy), refs.foreground(name)
            if i not in pack.test:
                train_rows.append(train_row(name, prediction, clean, noisy, foreground))
                continue
            row = test_row(name, prediction, alpha, clean, noisy, foreground)
            if classes is not None:
                B = prediction.shape[0]
                classes.add(refs.gt(name), prediction.clamp(0, 1).reshape(B, -1), {'clean': clean, 'noisy': noisy}, noisy)
            rows.append(row)
            print(name, {k: round(row[k]['unified']['hsi_psnr'], 3) for k in ('clean', 'noisy')},
                  {k: round(row[k]['unified']['hsi_sam_deg'], 3) for k in ('clean', 'noisy')}, flush=True)
    summary = summarize(rows, train_rows)
    if classes is not None:
        summary['per_class_test_pooled'] = classes.summary()
    v0 = pack.views[0]
    W, H = pack.width, pack.height
    return {'resolution_wh': [W, H], 'intrinsics': f"original {W}x{H} (fx={v0['fx']}, fy={v0['fy']}, cx={v0['cx']}, cy={v0['cy']})",
            'noisy_dataset': str(refs.noisy_dataset), 'clean_dataset': str(refs.clean_dataset), 'normalization_scale': refs.scale,
            'gt_foreground_rule': refs.rule,
            'test_views': [pack.names[i] for i in pack.test], 'calibration_views_exported': len(pack.train),
            'summary': summary, 'views': rows, 'train_views': train_rows,
            'render_seconds_per_view_mean': float(np.mean(render_seconds)),
            'metric_notes': {'unified': 'unified_hsi_metrics (metrics.py): PSNR over all pixels with peak 1 after '
                                        'x/scale and clamp [0,1]; SAM over pixels whose reference norm > 1e-6 (macro mean over TEST views)',
                             'gt_foreground': 'same quantities restricted to GT coverage > 0 (no coverage key: material_id below the '
                                              'light-source class, i.e. reflective pixels)' if args.foreground != 'surface' else
                                             'same quantities restricted to the released surface_evaluation_mask (coverage >= 0.5)',
                             'unclipped': 'no clamp of prediction or reference (scaled units, peak 1): psnr_all_pixels over all '
                                          'pixels; psnr / sam_deg / sam_p95_deg over the gt_foreground mask (SAM: reference norm > 1e-6)',
                             'train': 'unified metrics, macro mean over all TRAIN views (train/test gap check)'},
            'seconds': time.time() - t0}

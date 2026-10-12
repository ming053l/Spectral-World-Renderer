#!/usr/bin/env python3
"""Material identification for datasets whose material_class_protocol.json lists 'classes' (C first-interface
classes; Karst: limestone, green_vegetation, soil, wood_bark, water_surface), with the rule functions of benchmark.py:
  * one global many-to-one map channel -> class by majority vote of the dominant channel at the GT-foreground pixels
    (gt 'material' >= 0, equal to coverage > 0) of all TRAIN views (split_protocol.json), written (with its sha256)
    before any TEST prediction is read; never-dominant channels -> -1 (unassigned);
  * TEST: mapped dominant slot (and, separately, argmax of summed class abundance), pooled confusion over the TEST
    views, IoU per class, mIoU over classes with a non-empty union, pixel accuracy.
Label space 'classes<C>'. Diagnostics: mean IoU of the opaque classes next to the water IoU; area-abundance and
first-surface radiance-weighted abundance errors over all, opaque-labelled and water-labelled pixels.
Usage: python -m abundance_gs.scorers.protocol --dataset <dir> --predictions <dir> --output <json> [--endmembers <npy>]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from abundance_gs.scorers import benchmark as ebm


def read(pred, stem):
    with np.load(pred / f'{stem}_abundance.npz') as z:
        return ebm.normalize_abundance(z['abundance'], 'chw')[0]


def labels(ds, stem, C):
    with np.load(ds / 'gt' / f'{stem}.npz') as z:
        lab = z['material'].astype(np.int64)
        fg = z['coverage'] > 0
        area = np.moveaxis(z['abundance_area'][..., :C], -1, 0).astype(np.float32)
        rad = np.moveaxis(z['abundance_radiance'][..., :C], -1, 0).astype(np.float32)
    if not np.array_equal(lab >= 0, fg):
        raise ValueError('Material labels and explicit GT coverage foreground differ')
    return lab, area, rad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--endmembers', type=Path)
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    a = ap.parse_args()
    ds = a.dataset
    proto = json.loads((ds / 'material_class_protocol.json').read_text())
    names = list(proto['classes'])
    C = len(names)
    assert C == int(proto.get('class_count', C))
    opaque = [names.index(c) for c in proto.get('opaque_reference_endmember_classes', [])]
    nonopaque = [i for i in range(C) if i not in opaque]
    split = json.loads((ds / 'split_protocol.json').read_text())['frames']
    train, held = split['train'], split[a.split]
    assert not set(train) & set(split['val'] + split['test'])
    space = f'classes{C}'
    votes = mass = None
    for stem in train:
        ab = read(a.predictions, stem)
        lab, _, _ = labels(ds, stem, C)
        if votes is None:
            votes, mass = np.zeros((ab.shape[0], C), np.int64), np.zeros(ab.shape[0])
        mass += ab[:, lab >= 0].sum(1, dtype=np.float64)
        ebm.add_votes(votes, ab, lab)
    mapping = ebm.mapping_from_votes(votes)
    calib = {'algorithm': 'global many-to-one dominant-channel majority vote (evaluate_benchmark_materials.add_votes / '
                          'mapping_from_votes)', 'calibration_split': 'train', 'calibration_views': train,
             'held_out_gt_used_for_mapping': False, 'K': int(len(mapping)),
             'foreground_mask': "GT coverage > 0, verified equivalent to gt 'material' >= 0",
             'mapping_tie_rule': 'lowest class ID among tied positive votes',
             'channel_foreground_training_abundance_mass': mass.tolist(),
             'label_spaces': {space: {'channel_to_class': mapping.tolist(), 'votes': votes.tolist(),
                                      'unassigned_channels': np.flatnonzero(mapping < 0).tolist(),
                                      'fitted_on': 'all GT-foreground TRAIN pixels'}}}
    calib['sha256'] = hashlib.sha256(json.dumps(calib, sort_keys=True).encode()).hexdigest()
    a.output.with_suffix('.calibration.json').write_text(json.dumps(calib, indent=1) + '\n')   # frozen before TEST is read
    dom_cm, sum_cm = np.zeros((C, C + 1), np.int64), np.zeros((C, C + 1), np.int64)
    regions = {'all': None, 'opaque_labelled': opaque, 'water_or_nonopaque_labelled': nonopaque}
    err = {(t, r): {'sq': np.zeros(C), 'abs': np.zeros(C), 'n': 0} for t in ('area', 'radiance') for r in regions}
    pixels, unassigned_mass = 0, 0.
    for stem in held:
        ab = read(a.predictions, stem)
        lab, area, rad = labels(ds, stem, C)
        class_ab, unknown = ebm.aggregate_channels(ab, mapping, C)
        dominant, summed = ebm.mapped_labels(ab, mapping, class_ab, unknown)
        dom_cm += ebm.confusion(lab, dominant, C)
        sum_cm += ebm.confusion(lab, summed, C)
        v = lab >= 0
        pixels += int(v.sum())
        unassigned_mass += float(unknown[v].sum(dtype=np.float64))
        for r, ids in regions.items():
            m = v if ids is None else (v & np.isin(lab, ids))
            for t, tgt in (('area', area), ('radiance', rad)):
                d = (class_ab[:, m] - tgt[:, m]).astype(np.float64)
                e = err[(t, r)]
                e['sq'] += (d ** 2).sum(1)
                e['abs'] += np.abs(d).sum(1)
                e['n'] += int(m.sum())
    dm = ebm.confusion_metrics(dom_cm, names)
    sm = ebm.confusion_metrics(sum_cm, names)
    for m in (dm, sm):
        io = m['iou_per_class']
        m['opaque_class_mean_iou_diagnostic'] = (float(np.mean([io[i] for i in opaque if io[i] is not None]))
                                                 if opaque and any(io[i] is not None for i in opaque) else None)
        m['nonopaque_class_iou'] = {names[i]: io[i] for i in nonopaque}
    errors = {}
    for (t, r), e in err.items():
        n = max(e['n'], 1)
        errors.setdefault(t, {})[r] = {'pixels': e['n'], 'RMSE': float(np.sqrt(e['sq'].sum() / (n * C))),
                                       'MAE': float(e['abs'].sum() / (n * C)),
                                       'RMSE_per_class': np.sqrt(e['sq'] / n).tolist(), 'MAE_per_class': (e['abs'] / n).tolist()}
    res = {space: {'mapped_dominant_slot_material_agreement': dm, 'summed_class_abundance_material_agreement': sm,
                   'mean_unassigned_foreground_abundance_mass': unassigned_mass / max(pixels, 1),
                   'class_abundance_errors': errors,
                   'class_abundance_error_note': "'area' = gt abundance_area (first-interface footprint fraction); 'radiance' = gt "
                                                 'abundance_radiance (first-interface radiance-weighted); both group PRIMARY FIRST '
                                                 'INTERFACES (water includes its reflected/transmitted power). Not radiance_origin_power.'}}
    out = {'evaluator': 'score_protocol_materials.py (evaluate_benchmark_materials.py rule functions imported; that script only '
                        'supports the 6/8-class Garden schema)', 'dataset': str(ds), 'class_names': names,
           'opaque_classes': [names[i] for i in opaque], 'calibration': calib,
           'evaluation': {'split': a.split, 'views': held, 'mapping_refitted_on_scored_views': False, 'label_spaces': res}}
    if a.endmembers is not None:
        out['endmembers_radiance_file'] = str(a.endmembers)
        out['endmember_note'] = 'Water has no fixed endmember / SAM target (protocol water_endmember = null).'
    a.output.write_text(json.dumps(out, indent=1) + '\n')
    print(space, 'map', mapping.tolist(), 'acc %.3f mIoU %.3f' % (dm['foreground_pixel_accuracy'], dm['foreground_miou'] or 0),
          [None if v is None else round(v, 3) for v in dm['iou_per_class']])


if __name__ == '__main__':
    main()

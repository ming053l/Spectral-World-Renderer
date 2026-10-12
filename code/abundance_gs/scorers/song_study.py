#!/usr/bin/env python3
"""Material identification for the Study layout (gt/<stem>.npz with material_id 0..5, abundance_area and
desk_roi_area), with the rule functions of benchmark.py:
  * one global many-to-one map learned channel -> class by majority vote of the dominant channel at the scored pixels of
    all TRAIN views (transforms_train.json), fixed before TEST is read; never-dominant channels -> -1 (unassigned);
  * TEST: mapped dominant slot (and, separately, argmax of summed class abundance), pooled confusion over the TEST
    views, IoU per class, mIoU over classes with a non-empty union, pixel accuracy.
Label spaces (both use the single map fitted on reflective5 TRAIN pixels):
  reflective5  pixels with material_id in 0..4 (class 5, light-source geometry, excluded)
  desk_roi5    the same, restricted to the desk ROI (desk_roi_area >= 0.5)
Abundance errors vs the GT area occupancy (abundance_area[..., :5]) are reported as a diagnostic.
Usage: python -m abundance_gs.scorers.song_study --dataset <dir> --predictions <dir> --output <json> [--endmembers <npy>]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from abundance_gs.scorers import benchmark as ebm

NAMES = ['aged_reddish_brown_wood', 'pale_paper_proxy', 'carbon_black_ink_proxy', 'dark_stone_inkstone_proxy',
         'matte_ceramic_proxy']
C = 5
ROI_THRESHOLD = 0.5


def stems(ds, split):
    return [Path(f['file_path']).stem for f in json.loads((ds / f'transforms_{split}.json').read_text())['frames']]


def labels(ds, stem):
    with np.load(ds / 'gt' / f'{stem}.npz') as z:
        mid = z['material_id'].astype(np.int64)
        roi = z['desk_roi_area'] >= ROI_THRESHOLD
        area = np.moveaxis(z['abundance_area'][..., :C], -1, 0).astype(np.float32)
    refl = np.where(mid <= 4, mid, -1)
    return {'reflective5': refl, 'desk_roi5': np.where(roi, refl, -1)}, area


def read(pred, stem):
    with np.load(pred / f'{stem}_abundance.npz') as z:
        return ebm.normalize_abundance(z['abundance'], 'chw')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--endmembers', type=Path)
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    a = ap.parse_args()
    ds = a.dataset
    train, held = stems(ds, 'train'), stems(ds, a.split)
    assert not set(train) & set(held + stems(ds, 'val' if a.split == 'test' else 'test'))
    votes = mass = None
    for stem in train:
        ab = read(a.predictions, stem)
        lab, _ = labels(ds, stem)
        if votes is None:
            votes, mass = np.zeros((ab.shape[0], C), np.int64), np.zeros(ab.shape[0])
        mass += ab[:, lab['reflective5'] >= 0].sum(1, dtype=np.float64)
        ebm.add_votes(votes, ab, lab['reflective5'])
    mapping = ebm.mapping_from_votes(votes)
    calib = {'algorithm': 'global many-to-one dominant-channel majority vote (evaluate_benchmark_materials.add_votes / '
                          'mapping_from_votes)', 'calibration_split': 'train', 'calibration_views': train,
             'held_out_gt_used_for_mapping': False, 'K': int(len(mapping)),
             'scored_pixels': 'material_id in 0..4 (class 5 light-source geometry excluded)',
             'mapping_tie_rule': 'lowest class ID among tied positive votes', 'channel_foreground_training_abundance_mass': mass.tolist(),
             'label_spaces': {s: {'channel_to_class': mapping.tolist(), 'votes': votes.tolist(),
                                  'unassigned_channels': np.flatnonzero(mapping < 0).tolist(),
                                  'fitted_on': 'reflective5 TRAIN pixels'} for s in ('reflective5', 'desk_roi5')}}
    calib['sha256'] = hashlib.sha256(json.dumps(calib, sort_keys=True).encode()).hexdigest()
    st = {s: {'dominant': np.zeros((C, C + 1), np.int64), 'summed': np.zeros((C, C + 1), np.int64), 'sq': np.zeros(C), 'abs': np.zeros(C),
              'pixels': 0, 'unassigned_mass': 0.} for s in ('reflective5', 'desk_roi5')}
    for stem in held:
        ab = read(a.predictions, stem)
        lab, area = labels(ds, stem)
        class_ab, unknown = ebm.aggregate_channels(ab, mapping, C)
        dominant, summed = ebm.mapped_labels(ab, mapping, class_ab, unknown)
        for s, l in lab.items():
            st[s]['dominant'] += ebm.confusion(l, dominant, C)
            st[s]['summed'] += ebm.confusion(l, summed, C)
            v = l >= 0
            st[s]['pixels'] += int(v.sum())
            st[s]['unassigned_mass'] += float(unknown[v].sum(dtype=np.float64))
            d = (class_ab[:, v] - area[:, v]).astype(np.float64)
            st[s]['sq'] += (d ** 2).sum(1)
            st[s]['abs'] += np.abs(d).sum(1)
    res = {}
    for s, x in st.items():
        n = max(x['pixels'], 1)
        res[s] = {'mapped_dominant_slot_material_agreement': ebm.confusion_metrics(x['dominant'], NAMES),
                  'summed_class_abundance_material_agreement': ebm.confusion_metrics(x['summed'], NAMES),
                  'mean_unassigned_foreground_abundance_mass': x['unassigned_mass'] / n,
                  'class_abundance_errors': {'area': {'RMSE': float(np.sqrt(x['sq'].sum() / (n * C))), 'MAE': float(x['abs'].sum() / (n * C)),
                                                      'RMSE_per_class': np.sqrt(x['sq'] / n).tolist(), 'MAE_per_class': (x['abs'] / n).tolist()}}}
    out = {'evaluator': 'score_song_materials.py (evaluate_benchmark_materials.py rule functions imported; that script itself does '
                        'not support this dataset layout)', 'dataset': str(ds), 'calibration': calib,
           'evaluation': {'split': a.split, 'views': held, 'mapping_refitted_on_scored_views': False,
                          'desk_roi_rule': f'desk_roi_area >= {ROI_THRESHOLD}', 'label_spaces': res}}
    if a.endmembers is not None:
        out['endmembers_radiance_file'] = str(a.endmembers)
        out['endmember_note'] = ('No per-material radiance reference exists in this dataset (path-traced indoor light; '
                                 'spectral_materials.npz is reflectance), so no radiance-space endmember SAM is computed.')
    a.output.write_text(json.dumps(out, indent=1) + '\n')
    for s in res:
        m = res[s]['mapped_dominant_slot_material_agreement']
        print(s, 'map', mapping.tolist(), 'acc %.3f mIoU %.3f' % (m['foreground_pixel_accuracy'], m['foreground_miou'] or 0),
              [None if v is None else round(v, 3) for v in m['iou_per_class']])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""City scoring with previews: runs the merged-label-space scorer (merged.py), draws previews of the 2nd and the
middle TEST view with the protocol primary palette, saves the TEST dominant-slot maps (dominant_test.npz) and
deletes the exported abundance predictions unless --keep-predictions."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def protocol_previews(DS, ev, mapping, views, merged=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    proto = json.loads((DS / 'material_class_protocol.json').read_text())
    names = list(proto['primary_classes' if merged else 'classes'])
    pal = np.array(proto['primary_palette' if merged else 'palette_rgb'] + [[255, 0, 255]], np.uint8)   # last: unassigned
    C = len(names)
    out = ev / 'previews'
    out.mkdir(exist_ok=True)
    for stem in views:
        with np.load(ev / 'predictions' / f'{stem}_abundance.npz') as z:
            a = z['abundance'].astype(np.float32)
        total = a.sum(0)
        dom = np.where(total > 1e-12, mapping[a.argmax(0)], -1)
        pred = np.zeros((*dom.shape, 3), np.uint8)
        pred[dom >= 0] = pal[dom[dom >= 0]]
        pred[(dom < 0) & (total > 1e-12)] = pal[C]
        lab = np.load(DS / 'gt' / f'{stem}.npz')['material_merged' if merged else 'material']
        gtimg = np.zeros((*lab.shape, 3), np.uint8)
        gtimg[lab >= 0] = pal[lab[lab >= 0]]
        pred[lab < 0] = (pred[lab < 0] * 0.35).astype(np.uint8)            # GT background dimmed (not scored)
        rgb = np.asarray(Image.open(DS / 'images' / f'{stem}.png').convert('RGB'))
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
        for axis, im, title in zip(ax, (rgb, gtimg, pred), ('scientific RGB', f'GT material ({C} first-interface classes)',
                                                             f'predicted dominant channel -> frozen TRAIN map ({C} classes)')):
            axis.imshow(im)
            axis.set_title(f'{stem}: {title}', fontsize=8)
            axis.axis('off')
        handles = [plt.Rectangle((0, 0), 1, 1, color=pal[c] / 255) for c in range(C + 1)]
        fig.legend(handles, names + ['unassigned'], loc='lower center', ncol=C + 1, fontsize=7)
        fig.savefig(out / f'{stem}_dominant_class.png', dpi=110, bbox_inches='tight')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--keep-predictions', action='store_true')
    a = ap.parse_args()
    ev, DS = a.eval_dir, a.dataset
    pred = ev / 'predictions'
    out = ev / 'benchmark_material_evaluation.json'
    proto = json.loads((DS / 'material_class_protocol.json').read_text())
    if not (DS / 'split_protocol.json').exists() or len(proto.get('primary_classes', [])) in (0, 6):
        raise ValueError('scorers.city handles the merged City layout only')
    subprocess.run([sys.executable, '-m', 'abundance_gs.scorers.merged',
                    '--dataset', str(DS), '--predictions', str(pred), '--endmembers', str(ev / 'endmembers_radiance.npy'),
                    '--split', 'test', '--output', str(out)], check=True)
    mapping = np.array(json.loads(out.read_text())['calibration']['label_spaces'][f"classes{len(proto['primary_classes'])}"]['channel_to_class'])
    test = json.loads((DS / 'split_protocol.json').read_text())['frames']['test']
    protocol_previews(DS, ev, mapping, (test[1], test[len(test) // 2]), merged=True)
    # TEST dominant-slot maps (uint8 argmax over K; 255 = zero total abundance)
    doms = []
    for stem in test:
        with np.load(pred / f'{stem}_abundance.npz') as z:
            ab = z['abundance'].astype(np.float32)
        doms.append(np.where(ab.sum(0) > 1e-12, ab.argmax(0), 255).astype(np.uint8))
    np.savez_compressed(ev / 'dominant_test.npz', stems=np.array(test), dominant=np.stack(doms))
    n = len(list(pred.glob('*_abundance.npz')))
    size = sum(p.stat().st_size for p in pred.glob('*'))
    if not a.keep_predictions:
        shutil.rmtree(pred)
    (ev / 'score_done.json').write_text(json.dumps({'status': 'complete', 'evaluator_output': str(out),
                                                    'predictions_deleted': not a.keep_predictions, 'prediction_files': n,
                                                    'prediction_bytes': size}, indent=1) + '\n')


if __name__ == '__main__':
    main()

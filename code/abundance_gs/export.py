"""Stage `pack`: export cameras, split, RGB and HSI paths of a scene into a pack.

The scene is loaded from the saved args ('args' dict) of a reference state file with scene_loader.py; every HSI cube
is re-loaded from its recorded path and checked identical. Writes <output>/pack.json and rgb.npy (layout: data.py).
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import load_hsi_numpy, resize_hsi_tensor, resolve_hsi_path
from .scene_loader import load_scene


def _view(cam, split, saved):
    """pack.json entry and RGB [H, W, 3] of one camera; RGB is alpha-premultiplied where the image has alpha."""
    w, h = int(cam.image_width), int(cam.image_height)
    c2w = torch.linalg.inv(cam.world_view_transform.detach().cpu().double().T)   # world->camera is [R | t]
    hsi_path = resolve_hsi_path(cam.image_name, source_path=saved['source_path'], hsi_dir=saved['hsi_dir'])
    assert hsi_path is not None, cam.image_name
    again = resize_hsi_tensor(load_hsi_numpy(hsi_path), (w, h)).clamp(0.0, 1.0)
    image = cam.original_image.detach().cpu().float()
    alpha = cam.alpha_mask.detach().cpu().float() if cam.alpha_mask is not None else None
    premultiplied = alpha is not None and float(alpha.min()) < 1.0
    if premultiplied:
        image = image * alpha
    entry = {'name': str(cam.image_name), 'split': split, 'width': w, 'height': h,
             'fx': w / (2.0 * math.tan(float(cam.FoVx) / 2.0)), 'fy': h / (2.0 * math.tan(float(cam.FoVy) / 2.0)),
             'cx': w / 2.0, 'cy': h / 2.0, 'c2w': c2w.tolist(), 'hsi_path': str(hsi_path),
             'bands': int(cam.original_hsi.shape[0]),
             'hsi_reload_identical': bool(torch.equal(again, cam.original_hsi.detach().cpu().float())),
             'hsi_zero_fraction': float((cam.original_hsi.float().norm(dim=0) <= 1e-6).float().mean())}
    return entry, image.permute(1, 2, 0).numpy().astype(np.float32), premultiplied


def export_pack(state, scene_id, out, anchor=None, source_path=None, hsi_dir=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(state, map_location='cpu', weights_only=False)['args']
    if source_path:
        saved['source_path'] = source_path
        saved['hsi_dir'] = hsi_dir or str(Path(source_path) / 'hsi')
    anchor = Path(anchor) if anchor else Path(str(saved['model_path']))
    iteration = int(saved['iteration'])
    _, train_cameras, test_cameras = load_scene(saved, anchor)

    views, rgbs, alpha_used = [], [], False
    for split, cameras in (('train', train_cameras), ('test', test_cameras)):
        for cam in cameras:
            if getattr(cam, 'original_hsi', None) is None:
                continue
            entry, rgb, premultiplied = _view(cam, split, saved)
            views.append(entry)
            rgbs.append(rgb)
            alpha_used |= premultiplied
    sizes = {(v['width'], v['height']) for v in views}
    assert len(sizes) == 1, sizes
    np.save(out / 'rgb.npy', np.stack(rgbs))
    ply = anchor / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
    pack = {'scene_id': scene_id, 'views': views, 'bands': views[0]['bands'],
            'train': [v['name'] for v in views if v['split'] == 'train'],
            'test': [v['name'] for v in views if v['split'] == 'test'],
            'anchor_ply': str(ply), 'anchor_ply_sha256': hashlib.sha256(ply.read_bytes()).hexdigest(),
            'reference_state': str(Path(state).resolve()), 'source_path': str(saved['source_path']),
            'hsi_dir': str(saved['hsi_dir']), 'resolution_arg': saved.get('resolution'),
            'rgb_alpha_premultiplied': alpha_used, 'hsi_loader': 'data.load_hsi_numpy + resize_hsi_tensor + clamp(0,1)',
            'all_hsi_reload_identical': all(v['hsi_reload_identical'] for v in views),
            'camera_model': 'pinhole, principal point at the image centre (as the anchor and the canonical evaluator use it)'}
    (out / 'pack.json').write_text(json.dumps(pack, indent=1) + '\n')
    return pack


def run(args):
    pack = export_pack(args.state, args.scene_id, args.output, args.anchor, args.source_path, args.hsi_dir)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in pack.items() if k != 'views'}, indent=1))

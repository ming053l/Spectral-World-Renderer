"""Stage `anchor`: RGB 3DGS anchor with multi-view density control.

Runs the vendored FastGS trainer (third_party/fastgs/train.py) in this process with the standard 30k schedule, RGB
seed 0, SH degree 3, --densification_interval 500 --grad_abs_thresh 0.0012, then scores the saved ply on the TRAIN
and held-out TEST cameras of the training scene (images loaded once) with the 3DGS rasterizer (third_party/gs_rasterizer).
Writes <output>/point_cloud/iteration_<N>/point_cloud.ply and <output>/anchor.json.
"""
import contextlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from . import records, third_party
from .defaults import Anchor

D = Anchor()
TRAINER = third_party.HERE / 'fastgs'


def trainer_arguments(source, output, resolution, iterations, seed):
    it = str(iterations)
    return ['-s', source, '-m', str(output), '-r', resolution, '--data_device', 'cpu', '--eval', '--iterations', it,
            '--seed', str(seed), '--test_iterations', it, '--save_iterations', it,
            '--densification_interval', str(D.densification_interval), '--grad_abs_thresh', str(D.grad_abs_thresh),
            '--checkpoint_iterations', '-1']


def _fastgs():
    """The vendored FastGS trainer module (its packages `scene`, `utils`, ... importable from third_party/fastgs)."""
    third_party.use()
    if str(TRAINER) not in sys.path:
        sys.path.insert(0, str(TRAINER))
    spec = importlib.util.spec_from_file_location('fastgs_train', TRAINER / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_anchor(arguments, log):
    """Run the FastGS trainer in this process, its console output to `log`; returns (wall seconds, log text, scene)."""
    module = _fastgs()
    begin = time.time()
    with log.open('w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        scene = module.main(arguments)
    return time.time() - begin, log.read_text(), scene


def render_rgb(camera, gaussians, background):
    """3DGS render of the SH colours (no antialiasing)."""
    import torch
    from gs_rasterizer import GaussianRasterizationSettings, GaussianRasterizer
    means2D = torch.zeros_like(gaussians.get_xyz, requires_grad=True, device='cuda') + 0
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5), bg=background, scale_modifier=1.,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform, sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center, prefiltered=False, debug=False, antialiasing=False)
    image, _, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=gaussians.get_xyz, means2D=means2D, shs=gaussians.get_features, colors_precomp=None,
        opacities=gaussians.get_opacity, scales=gaussians.get_scaling, rotations=gaussians.get_rotation, cov3D_precomp=None)
    return image


def _reader_order(cameras):
    """Cameras in the order of the scene reader (the training scene shuffles them)."""
    return sorted(cameras, key=lambda c: (c.colmap_id, c.image_name))


def score_anchor(scene, ply):
    """Mean RGB PSNR of the saved ply on the TRAIN and held-out TEST cameras of the training scene."""
    import numpy as np
    import torch
    from scene import GaussianModel
    from utils.image_utils import psnr
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(ply))
    background = torch.zeros(3, device='cuda')
    scores = {}
    with torch.no_grad():
        for split, cameras in (('train', scene.getTrainCameras()), ('test', scene.getTestCameras())):
            values = [float(psnr(torch.clamp(render_rgb(cam, gaussians, background), 0, 1),
                                 torch.clamp(cam.original_image.cuda(), 0, 1)).mean()) for cam in _reader_order(cameras)]
            scores[split] = float(np.mean(values))
    return scores, int(gaussians.get_xyz.shape[0])


def run(a):
    source = str(Path(a.scene_dir).resolve())
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, config = records.expected('anchor', a)
    arguments = trainer_arguments(source, output, a.resolution, a.iterations, a.seed)
    wall, text, scene = train_anchor(arguments, output / 'train.log')
    train_time = re.findall(r'Training time: ([0-9.]+)', text)
    console = re.findall(r'Gaussian number: (\d+)', text)
    ply = output / 'point_cloud' / f'iteration_{a.iterations}' / 'point_cloud.ply'
    scores, primitives = score_anchor(scene, ply)
    out = {'status': 'complete', 'trainer_arguments': arguments, 'iterations': a.iterations, 'seed': a.seed,
           'wall_seconds': wall, 'train_seconds': float(train_time[-1]) if train_time else None,
           'primitives': primitives, 'console_gaussians': int(console[-1]) if console else None,
           'rgb_psnr_test': scores['test'], 'rgb_psnr_train': scores['train'], 'ply': str(ply), 'ply_bytes': ply.stat().st_size,
           'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}
    (output / 'anchor.json').write_text(json.dumps(out, indent=1) + '\n')
    records.complete(output, 'anchor', inputs, config, ply=str(ply), primitives=primitives, rgb_psnr_test=scores['test'])
    print(json.dumps({k: out[k] for k in ('primitives', 'train_seconds', 'rgb_psnr_test')}))

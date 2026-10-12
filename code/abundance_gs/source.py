"""Stage `source`: anchor Gaussians -> harmonic-texture (NHT) feature field, RGB feature refinement (paper Stage 1).

The anchor's means, scales, rotations and opacities become the NHT primitives; each scale is dilated by the
pixel-footprint variance that stands in for the anchor rasteriser's 0.3 px^2 low-pass filter; the 48 features per
Gaussian (4 tetrahedron vertices x 12 harmonic dims) start random. 5,000 steps of the NHT trainer (nht/trainer.py) on
the RGB companions train features and geometry (upstream learning rates, no density control, no position noise, no
opacity / scale regulariser). Cameras, RGB and split come from the pack; the world frame is the scene's own.
Writes <output>/ckpts/ckpt_<steps-1>_rank0.pt (the decoder's --source) and <output>/run.json.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import records
from .data import Pack
from .defaults import Source
from .nht.anchor_frame import load_anchor_ply, pixel_footprint_variance, to_nht_frame
from .nht.trainer import Config, Runner, create_splats_with_optimizers

D = Source()


class PackParser:
    """The scene description the NHT trainer reads: the pack and the scene scale (largest camera distance from the
    mean camera centre). The world frame is left as it is."""

    def __init__(self, pack):
        self.pack = pack
        centers = np.stack([pack.c2w(i).numpy() for i in range(len(pack.views))]).astype(np.float64)[:, :3, 3]
        self.scene_scale = float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))


class PackDataset:
    """The NHT trainer's Dataset over a pack's RGB TRAIN or TEST views."""

    def __init__(self, parser, split='train'):
        self.pack = parser.pack
        self.indices = np.asarray(self.pack.train if split == 'train' else self.pack.test)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        image = torch.from_numpy(np.ascontiguousarray(self.pack.rgb[i])).float() * 255.0
        return {'K': self.pack.K(i), 'camtoworld': self.pack.c2w(i), 'image': image, 'image_id': item, 'camera_idx': 0}


class CachedDataset(torch.utils.data.Dataset):
    """Every sample of a dataset decoded once and kept in memory."""

    def __init__(self, source):
        self.source = source
        self.samples = [source[i] for i in range(len(source))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def __getattr__(self, k):
        return getattr(self.source, k)


def dilated_anchor(ply, pack, train_views):
    """Anchor Gaussians in the NHT frame with scales widened by the pixel-footprint variance over TRAIN cameras."""
    anchor = to_nht_frame(load_anchor_ply(ply), np.eye(4))
    extra = pixel_footprint_variance(anchor['means'], [pack.c2w(i) for i in train_views], [pack.K(i) for i in train_views],
                                     pack.width, pack.height)
    sigma = anchor['scales'].exp()
    anchor['scales'] = torch.log(torch.sqrt(sigma ** 2 + extra[:, None]))
    return anchor, extra, sigma


def anchor_initializer(pack, anchor_info):
    """Splat initializer of the trainer: the dilated anchor Gaussians (means, scales, rotations, opacities)."""
    def init_splats(feature_module, cfg, scene_scale):
        anchor, extra, sigma = dilated_anchor(pack.meta['anchor_ply'], pack, pack.train)
        anchor_info.update(dilation='0.3 px^2 pixel-footprint variance, median over training cameras',
                           dilation_sigma_over_max_scale_median=float((extra.sqrt() / sigma.max(-1).values).median()),
                           unseen_by_training_cameras=int((extra == 0).sum()))
        splats, optimizers = create_splats_with_optimizers(anchor['means'].numpy(), feature_module, cfg, scene_scale)
        with torch.no_grad():
            for name in ('scales', 'quats', 'opacities'):
                splats[name].copy_(anchor[name].to(splats[name].device))
        anchor_info.update(primitives=len(anchor['means']))
        return splats, optimizers
    return init_splats


def finish(output, step):
    """Keep the last validation render only."""
    for path in (output / 'renders').glob('*.png'):
        if path.name != f'val_step{step}_0000.png':
            path.unlink()


def run(args):
    torch.set_num_threads(4)
    if (args.output / 'run.json').exists():
        raise FileExistsError(f'Use a fresh output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    pack = Pack(args.pack)
    if args.anchor_ply:
        pack.meta['anchor_ply'] = str(args.anchor_ply)
    if not pack.meta.get('anchor_ply'):
        raise ValueError('The pack records no anchor ply; pass --anchor-ply')
    inputs, config = records.expected('source', args)
    status = {'status': 'initializing', 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'scene_id': pack.meta['scene_id'], 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'feature_dim': D.feature_dim,
              'anchor_ply': pack.meta['anchor_ply']}

    def write():
        tmp = args.output / 'run.json.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(args.output / 'run.json')
    write()
    try:
        anchor_info = {}
        cfg = Config.for_steps(str(args.output), args.steps)
        parser = PackParser(pack)
        trainset, valset = PackDataset(parser, 'train'), PackDataset(parser, 'val')
        runner = Runner(cfg, parser, trainset, valset, anchor_initializer(pack, anchor_info), seed=args.seed)
        status.update(anchor=anchor_info, initial_primitives=len(runner.splats['means']), scene_scale=float(runner.scene_scale),
                      train_views=len(trainset), test_views=len(valset), width=pack.width, height=pack.height)
        runner.trainset, runner.valset = CachedDataset(trainset), CachedDataset(valset)
        (args.output / 'shader_config.json').write_text(json.dumps(runner.shader_config, indent=2) + '\n')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        status['status'] = 'training'
        write()
        start = time.perf_counter()
        runner.train()
        torch.cuda.synchronize()
        step = cfg.max_steps - 1
        train_stats = json.loads((args.output / f'stats/train_step{step:04d}_rank0.json').read_text())
        val_stats = json.loads((args.output / f'stats/val_step{step:04d}.json').read_text())
        ckpt = args.output / f'ckpts/ckpt_{step}_rank0.pt'
        status.update(status='complete', steps=cfg.max_steps, train_and_eval_seconds=time.perf_counter() - start,
                      train_seconds=train_stats['ellipse_time'], primitives=len(runner.splats['means']),
                      gpu_name=torch.cuda.get_device_name(), peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                      rgb_psnr=val_stats['psnr'], rgb_ssim=val_stats['ssim'], rgb_lpips=val_stats['lpips'],
                      checkpoint=str(ckpt), checkpoint_bytes=ckpt.stat().st_size)
        finish(args.output, step)
        records.complete(args.output, 'source', inputs, config, checkpoint=str(ckpt))
        write()
        print(json.dumps({k: status[k] for k in ('primitives', 'train_seconds', 'rgb_psnr', 'checkpoint')}))
    except Exception as error:
        status.update(status='failed', error=repr(error))
        write()
        raise

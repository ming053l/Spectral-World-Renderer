# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The harmonic-texture (NHT) source-field trainer of stage `source`.

Adapted from neural-harmonic-textures aov/examples/simple_trainer_nht_aov.py (Apache-2.0, third_party/THIRD_PARTY.md),
reduced to the configuration used here: RGB only, batch 1, single GPU, no density control, 3DGUT rasterization with
eval3d, fused-SSIM + L1 loss, cosine-decayed feature / MLP learning rates, the last 1/10 of the steps as colour
refinement with frozen geometry, EMA of the deferred MLP for evaluation. The order of operations and random draws
follows the upstream trainer. Primitives come from `init_splats` (the anchor) instead of an SfM cloud.
"""
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from .. import third_party
from ..defaults import Source
from .tcnn import configure_tcnn


@dataclass
class Config:
    result_dir: str
    max_steps: int = 5000                # 30k-step upstream schedule scaled by steps / 30000
    color_refine_steps: int = 500        # last steps: geometry and opacities frozen (3000 * steps / 30000)
    eval_step: int = 5000                # evaluation (EMA) and checkpoint after this many steps
    feature_dim: int = Source.feature_dim
    init_opa: float = .5
    ssim_lambda: float = .2
    near_plane: float = .01
    far_plane: float = 1e10
    tile_size: int = 8
    means_lr: float = 1.6e-4
    scales_lr: float = 4e-3
    opacities_lr: float = 4e-2
    quats_lr: float = 8e-4
    features_lr: float = 15e-3
    features_lr_final: float = .1         # cosine decay to this fraction (features and MLP)
    mlp_lr: float = 72e-5
    mlp_hidden_dim: int = 128
    mlp_num_layers: int = 3
    view_sh_degree: int = 3
    view_sh_scale: float = 3.
    ema_decay: float = .95
    lpips_net: str = 'vgg'

    @classmethod
    def for_steps(cls, result_dir, steps):
        factor = steps / 30000
        return cls(result_dir=result_dir, max_steps=int(30000 * factor), color_refine_steps=int(3000 * factor),
                   eval_step=int(30000 * factor))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_splats_with_optimizers(points, feature_module, cfg, scene_scale, device='cuda'):
    """Gaussians at `points` with the upstream initial draws (quaternions, features) and fused Adam optimisers."""
    rgbs = torch.from_numpy(np.zeros((len(points), 3), np.uint8) / 255.0).float()
    points = torch.from_numpy(points).float()
    scales = torch.zeros(len(points), 3)          # replaced by the anchor scales
    quats = torch.rand((len(points), 4))          # random draw kept: the features are drawn after it
    opacities = torch.logit(torch.full((len(points),), cfg.init_opa))
    params = [('means', torch.nn.Parameter(points), cfg.means_lr * scene_scale),
              ('scales', torch.nn.Parameter(scales), cfg.scales_lr),
              ('quats', torch.nn.Parameter(quats), cfg.quats_lr),
              ('opacities', torch.nn.Parameter(opacities), cfg.opacities_lr)]
    feature_module.init_features(params, rgbs)
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizers = {name: torch.optim.Adam([{'params': splats[name], 'lr': lr, 'name': name}], eps=1e-15,
                                         betas=(0.9, 0.999), fused=True)
                  for name, _, lr in params}
    return splats, optimizers


class Runner:
    """Source-field training on a dataset with `trainset` / `valset` samples {K, camtoworld, image, image_id}."""

    def __init__(self, cfg, parser, trainset, valset, init_splats, seed=42):
        third_party.use()
        configure_tcnn()
        from gsplat.nht.deferred_shader import HarmonicFeatures, DeferredShaderModuleAOV
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        set_random_seed(seed)
        self.cfg, self.device = cfg, 'cuda:0'
        self.ckpt_dir, self.stats_dir, self.render_dir = (f'{cfg.result_dir}/{d}' for d in ('ckpts', 'stats', 'renders'))
        for directory in (self.ckpt_dir, self.stats_dir, self.render_dir):
            os.makedirs(directory, exist_ok=True)
        self.parser, self.trainset, self.valset = parser, trainset, valset
        self.scene_scale = parser.scene_scale * 1.1
        feature_module = HarmonicFeatures(feature_dim=cfg.feature_dim, feature_lr=cfg.features_lr,
                                          features_init_min=-np.pi / 2.0, features_init_max=np.pi / 2.0)
        self.splats, self.optimizers = init_splats(feature_module, cfg, self.scene_scale)
        # deferred shader: RGB only (no auxiliary heads)
        self.deferred_module = DeferredShaderModuleAOV(
            feature_dim=cfg.feature_dim, enable_view_encoding=True, auxiliary_output_dim=0, view_encoding_type='sh',
            mlp_hidden_dim=cfg.mlp_hidden_dim, mlp_num_layers=cfg.mlp_num_layers, sh_degree=cfg.view_sh_degree,
            sh_scale=cfg.view_sh_scale, fourier_num_freqs=4, primitive_type='3dgs', center_ray_encoding=False,
            split_rgb_head=False, fused_tcnn_sigmoid=False, rgb_only_tcnn_sigmoid=True).to(self.device)
        self.deferred_optimizer = torch.optim.Adam(self.deferred_module.parameters(), lr=cfg.mlp_lr, weight_decay=0.0)
        self._ema_shadow = {n: p.data.clone() for n, p in self.deferred_module.named_parameters()}
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=cfg.lpips_net, normalize=True).to(self.device)
        self.lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(self.device)

    # ---- EMA of the deferred MLP (evaluation uses it)
    @torch.no_grad()
    def _update_ema(self):
        for n, p in self.deferred_module.named_parameters():
            self._ema_shadow[n].lerp_(p.data, 1.0 - self.cfg.ema_decay)

    @torch.no_grad()
    def _apply_ema(self):
        self._ema_saved = {n: p.data.clone() for n, p in self.deferred_module.named_parameters()}
        for n, p in self.deferred_module.named_parameters():
            p.data.copy_(self._ema_shadow[n])

    @torch.no_grad()
    def _restore_from_ema(self):
        for n, p in self.deferred_module.named_parameters():
            p.data.copy_(self._ema_saved[n])
        self._ema_saved = None

    def rasterize_splats(self, camtoworlds, Ks, width, height):
        from gsplat.nht import NHTParams
        from gsplat.rendering import rasterization
        cfg = self.cfg
        render_colors, render_alphas, info = rasterization(
            means=self.splats['means'], quats=self.splats['quats'], scales=torch.exp(self.splats['scales']),
            opacities=torch.sigmoid(self.splats['opacities']), colors=self.splats['features'], sh_degree=None,
            viewmats=torch.linalg.inv(camtoworlds), Ks=Ks, width=width, height=height, tile_size=cfg.tile_size,
            packed=False, absgrad=False, sparse_grad=False, rasterize_mode='classic', render_mode='RGB',
            distributed=False, camera_model='pinhole', with_ut=True, with_eval3d=True,
            nht_params=NHTParams(enabled=True, center_ray_mode=False, ray_dir_scale=self.deferred_module.ray_dir_scale),
            near_plane=cfg.near_plane, far_plane=cfg.far_plane)
        rgb_raw, _, extras = self.deferred_module(render_colors)
        if extras is not None:
            rgb_raw = torch.cat([rgb_raw, extras], dim=-1)
        return rgb_raw, render_alphas, info

    def schedulers(self):
        cfg = self.cfg
        return [torch.optim.lr_scheduler.ExponentialLR(self.optimizers['means'], gamma=0.01 ** (1.0 / cfg.max_steps)),
                torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers['features'], T_max=cfg.max_steps,
                                                           eta_min=cfg.features_lr * cfg.features_lr_final),
                torch.optim.lr_scheduler.CosineAnnealingLR(self.deferred_optimizer, T_max=cfg.max_steps,
                                                           eta_min=cfg.mlp_lr * cfg.features_lr_final)]

    def train(self):
        from fused_ssim import fused_ssim
        from gsplat.nht.exporter import cast_state_dict_to_fp16
        cfg, device = self.cfg, self.device
        schedulers = self.schedulers()
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=0,
                                                  persistent_workers=False, pin_memory=True)
        trainloader_iter = iter(trainloader)
        color_refine_start = cfg.max_steps - cfg.color_refine_steps if cfg.color_refine_steps > 0 else cfg.max_steps
        in_color_refine = False
        global_tic = time.time()
        for step in range(cfg.max_steps):
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            camtoworlds = data['camtoworld'].to(device)
            Ks = data['K'].to(device)
            pixels = data['image'].to(device) / 255.0
            height, width = pixels.shape[1:3]
            if not in_color_refine and step >= color_refine_start:
                in_color_refine = True
                self._freeze_geometry()
                print(f'[step {step}] colour refinement: geometry and opacities frozen', flush=True)
            colors, _, _ = self.rasterize_splats(camtoworlds, Ks, width, height)
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding='valid')
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            loss.backward()
            if step == cfg.eval_step - 1 or step == cfg.max_steps - 1:      # checkpoint before the update
                self._save(step, global_tic, cast_state_dict_to_fp16)
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            self.deferred_optimizer.step()
            self.deferred_optimizer.zero_grad(set_to_none=True)
            self._update_ema()
            for scheduler in schedulers:
                scheduler.step()
            if in_color_refine:
                self._freeze_geometry()
            if step == cfg.eval_step - 1:
                self.eval(step)

    def _freeze_geometry(self):
        for name in ('means', 'scales', 'quats', 'opacities'):
            for group in self.optimizers[name].param_groups:
                group['lr'] = 0.0

    def _save(self, step, global_tic, cast_state_dict_to_fp16):
        stats = {'mem': torch.cuda.max_memory_allocated() / 1024 ** 3, 'ellipse_time': time.time() - global_tic,
                 'num_GS': len(self.splats['means'])}
        print('Step: ', step, stats, flush=True)
        with open(f'{self.stats_dir}/train_step{step:04d}_rank0.json', 'w') as f:
            json.dump(stats, f)
        splats_state = dict(self.splats.state_dict())
        splats_state['features'] = splats_state['features'].detach().to(torch.float16)      # fp16, as the NHT kernels read them
        data = {'step': step, 'splats': splats_state,
                'aov_config': {'lseg_feature_dim': 0, 'dinov3_feature_dim': 0, 'rgb2x_channels': None},
                'deferred_module': cast_state_dict_to_fp16(self.deferred_module.state_dict(), prefix='backbone.'),
                'deferred_ema': cast_state_dict_to_fp16(self._ema_shadow, prefix='backbone.')}
        torch.save(data, f'{self.ckpt_dir}/ckpt_{step}_rank0.pt')

    @torch.no_grad()
    def eval(self, step, stage='val'):
        """RGB PSNR / SSIM / LPIPS of the EMA model on the TEST views; writes renders and stats."""
        self._apply_ema()
        device = self.device
        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)
        warmup = next(iter(valloader))
        for _ in range(5):                       # warm-up renders
            self.rasterize_splats(warmup['camtoworld'].to(device), warmup['K'].to(device),
                                  warmup['image'].shape[2], warmup['image'].shape[1])
            torch.cuda.synchronize()
        ellipse_time, metrics = 0, defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds, Ks = data['camtoworld'].to(device), data['K'].to(device)
            pixels = data['image'].to(device) / 255.0
            height, width = pixels.shape[1:3]
            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(camtoworlds, Ks, width, height)
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(f'{self.render_dir}/{stage}_step{step}_{i:04d}.png', (canvas * 255).astype(np.uint8))
            pixels_p, colors_p = pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            metrics['psnr'].append(self.psnr(colors_p, pixels_p))
            metrics['ssim'].append(self.ssim(colors_p, pixels_p))
            metrics['lpips'].append(self.lpips(colors_p, pixels_p))
            metrics['lpips_alex'].append(self.lpips_alex(colors_p, pixels_p))
        ellipse_time /= len(valloader)
        with open(f'{self.stats_dir}/{stage}_step{step:04d}_per_image.json', 'w') as f:
            json.dump({k: [x.item() for x in v] for k, v in metrics.items()}, f)
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(ellipse_time=ellipse_time, num_GS=len(self.splats['means']))
        print(f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
              f"Time: {stats['ellipse_time']:.3f}s/image Number of GS: {stats['num_GS']}", flush=True)
        with open(f'{self.stats_dir}/{stage}_step{step:04d}.json', 'w') as f:
            json.dump(stats, f)
        self._restore_from_ema()

    @property
    def shader_config(self):
        dm = self.deferred_module
        return dict(feature_dim=dm.feature_dim, enable_view_encoding=dm.enable_view_encoding,
                    view_encoding_type=dm.view_encoding_type, mlp_hidden_dim=dm.mlp_hidden_dim,
                    mlp_num_layers=dm.mlp_num_layers, sh_degree=dm.view_sh_degree, sh_scale=dm.sh_scale,
                    fourier_num_freqs=dm.fourier_num_freqs, primitive_type=dm.primitive_type,
                    center_ray_encoding=dm.center_ray_encoding, lseg_feature_dim=0)

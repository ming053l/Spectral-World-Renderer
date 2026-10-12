"""Rendering a decoder over the views of a pack: shared by train, eval, render and fps.

A `ViewRenderer` renders whole views (reference FP32, the fast renderer, or the fast renderer with fused kernels) and
memoizes camera centres; the helpers turn its flat outputs into cubes / maps, time renders, and average metric rows.
"""
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .data import Pack
from .defaults import Render
from .model import Hits, load_model, render_view, render_view_fast, render_view_fused


class ViewRenderer:
    """renderer(i) -> spectrum [H*W, B], abundance [H*W, K], alpha [H*W, 1] of pack view i.

    fast: False (reference), True (fast renderer) or 'fused' (fast renderer with the fused CUDA kernels).
    """

    def __init__(self, model, hits, pack, fast=False, tau=Render.fast_tau):
        self.model, self.hits, self.pack, self.fast, self.tau = model, hits, pack, fast, tau
        self.device = model.features.device
        self.centers = {}

    def center(self, i):
        if i not in self.centers:
            self.centers[i] = self.pack.center(i, self.device)
        return self.centers[i]

    def __call__(self, i):
        if self.fast == 'fused':
            return render_view_fused(self.model, self.hits, i, self.center(i), self.tau)
        if self.fast:
            return render_view_fast(self.model, self.hits, i, self.center(i), self.tau)
        return render_view(self.model, self.hits, i, self.center(i))

    @property
    def description(self):
        if self.fast == 'fused':
            return f'fast fp16 (fused kernels), skip weight < {self.tau:g}'
        return f'fast fp16, skip weight < {self.tau:g}' if self.fast else 'reference fp32'


def open_model(model_path, pack_path, fast=False, tau=Render.fast_tau, device='cuda'):
    """Trained model_ema.pt on a pack: (renderer, model, splats, pack, hits); hit lists are traced per view, not cached."""
    pack = Pack(pack_path)
    model, splats = load_model(model_path, device)
    hits = Hits(splats, pack, 0)
    return ViewRenderer(model, hits, pack, fast, tau), model, splats, pack, hits


def cube(flat, pack):
    """[H*W, C] -> [C, H, W]."""
    return flat.T.reshape(-1, pack.height, pack.width)


def abundance_npz(abundance, alpha, pack, stride=1, clamp=True):
    """float16 abundance [K, H, W] and alpha [H, W] (spatially subsampled by `stride`) for np.savez_compressed."""
    maps = cube(abundance, pack)[:, ::stride, ::stride]
    if clamp:
        maps = maps.clamp_min(0)
    return dict(abundance=maps.half().cpu().numpy(),
                alpha=alpha.reshape(pack.height, pack.width)[::stride, ::stride].half().cpu().numpy())


def timed(function, *args):
    """(result, seconds) of a GPU call, synchronized on both sides."""
    torch.cuda.synchronize()
    tic = time.perf_counter()
    result = function(*args)
    torch.cuda.synchronize()
    return result, time.perf_counter() - tic


def render_times(renderer, views, repeats=Render.fps_repeats):
    """Seconds per render over `repeats` passes of `views`; the first pass (warm-up) is discarded."""
    times = []
    for repeat in range(repeats):
        for i in views:
            _, seconds = timed(renderer, i)
            if repeat:
                times.append(seconds)
    return times


def macro_mean(rows, keys=None):
    """Mean over rows of every float field of the first row (or of `keys`)."""
    keys = keys or [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


class BackgroundWriter:
    """File writes (e.g. np.savez_compressed) in worker threads, at most `pending` in flight, so that compression
    overlaps rendering. Leaving the context waits for every write and re-raises the first error."""

    def __init__(self, workers=2, pending=4):
        self.pool, self.pending, self.futures = ThreadPoolExecutor(workers), pending, deque()

    def submit(self, function, *args, **kwargs):
        while len(self.futures) >= self.pending:
            self.futures.popleft().result()
        self.futures.append(self.pool.submit(function, *args, **kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *error):
        try:
            while self.futures:
                self.futures.popleft().result()
        finally:
            self.pool.shutdown(wait=True)


def safe_name(name):
    return ''.join(c if c.isalnum() or c in '_.-' else '_' for c in name)

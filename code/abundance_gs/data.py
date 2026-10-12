"""Scene pack reader: the neutral hand-off between the canonical scene loader and every later stage.

A pack directory holds
  pack.json  scene_id, bands, and per view: name, split (train/test), width, height, fx, fy, cx, cy,
             camera-to-world (OpenCV axes, the scene's own world frame), hsi_path; plus anchor_ply and provenance
  rgb.npy    [V, H, W, 3] float32 in [0, 1], the RGB companions (alpha-premultiplied where the image has alpha)
HSI cubes are not copied: `Pack.hsi` reloads them from hsi_path with the canonical loader's functions (below).
Packs are written by stage `pack` (export.py).
"""
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class Pack:
    """A scene's views. Every later stage addresses a view by its index i into `views`: camera K(i) / c2w(i) /
    center(i), RGB rgb[i], HSI hsi(i), name names[i], split membership train / test. A per-view field added to
    pack.json (e.g. a capture time) is available as views[i][field] without further changes here."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'pack.json').read_text())
        self.views = self.meta['views']
        self.names = [v['name'] for v in self.views]
        self.train = [i for i, v in enumerate(self.views) if v['split'] == 'train']
        self.test = [i for i, v in enumerate(self.views) if v['split'] == 'test']
        self.width, self.height = self.views[0]['width'], self.views[0]['height']
        self.bands = self.meta['bands']
        self._rgb = None

    @property
    def pixels(self):
        return self.width * self.height

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = np.load(self.directory / 'rgb.npy', mmap_mode='r')
        return self._rgb

    def K(self, i):
        v = self.views[i]
        return torch.tensor([[v['fx'], 0, v['cx']], [0, v['fy'], v['cy']], [0, 0, 1]], dtype=torch.float32)

    def c2w(self, i):
        return torch.tensor(self.views[i]['c2w'], dtype=torch.float64).float()

    def center(self, i, device='cuda'):
        """Camera centre of view i (the view direction of the camera SH adapter)."""
        return self.c2w(i)[:3, 3].to(device)

    def hsi(self, i):
        """[B, H, W] float32 in [0, 1], loaded exactly as the canonical loader does."""
        cube = load_hsi_numpy(self.views[i]['hsi_path'])
        return resize_hsi_tensor(cube, (self.width, self.height)).clamp(0.0, 1.0)

    def hsi_iter(self, views, workers=4):
        """(i, self.hsi(i)) in the order of `views`, at most `workers` cubes loaded ahead in threads."""
        return prefetch(views, self.hsi, workers)

    def select(self, spec):
        """View indices of 'test', 'train', 'all' (TEST then TRAIN) or a comma list of view names."""
        groups = {'test': self.test, 'train': self.train, 'all': self.test + self.train}
        if spec in groups:
            return list(groups[spec])
        index = {name: i for i, name in enumerate(self.names)}
        missing = [name for name in spec.split(',') if name not in index]
        if missing:
            raise KeyError(f'Views not in the pack: {missing}')
        return [index[name] for name in spec.split(',')]

    def limit_hsi_training(self, train_max_views=0, val_every_n_train=0):
        """Restrict `train` to the HSI-supervision subset; returns (all RGB TRAIN views, held-out validation views)."""
        rgb_train = list(self.train)
        self.train = hsi_training_views(self, train_max_views, val_every_n_train)
        return rgb_train, [i for i in rgb_train if i not in self.train]


def prefetch(items, load, workers=4):
    """(item, load(item)) in the order of `items`, at most `workers` loads ahead in threads (file reads overlap the
    caller's work)."""
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        for item in items:
            pending.append((item, pool.submit(load, item)))
            if len(pending) > workers:
                done, result = pending.popleft()
                yield done, result.result()
        while pending:
            done, result = pending.popleft()
            yield done, result.result()


def hsi_training_views(pack, train_max_views=0, val_every_n_train=0):
    """The decoder's HSI-supervision subset of the TRAIN views, chosen before any spectrum is read.

    train_max_views > 0 keeps the first N TRAIN views; val_every_n_train = n > 1 then holds out every n-th of them
    (validation views). Anchor and source field always use all RGB TRAIN views.
    """
    if train_max_views < 0 or val_every_n_train < 0:
        raise ValueError('TRAIN-subset limits must be nonnegative')
    selected = list(pack.train)
    if train_max_views > 0:
        selected = selected[:train_max_views]
    if val_every_n_train > 1 and len(selected) >= val_every_n_train:
        kept = [v for index, v in enumerate(selected) if (index + 1) % val_every_n_train != 0]
        if kept and len(kept) < len(selected):
            selected = kept
    if not selected or any(pack.views[i]['split'] != 'train' for i in selected):
        raise ValueError('A nonempty actual TRAIN supervision subset is required')
    return selected


# ---- canonical HSI loader (the cube reader and resize used for all methods in the paper)

def resolve_hsi_path(image_path, source_path=None, hsi_dir=None):
    image_path = Path(image_path)
    stem = image_path.stem

    candidates = []
    if hsi_dir:
        hsi_root = Path(hsi_dir)
        candidates.extend([hsi_root / f"{stem}.npy", hsi_root / f"{stem}.h5", hsi_root / f"{stem}.hdf5"])

    if source_path:
        source_path = Path(source_path)
        parent = source_path.parent
        candidates.extend([
            parent / f"{source_path.name}_reconstructed_npy" / f"{stem}.npy",
            source_path / "hsi" / f"{stem}.npy",
            source_path / "truck_hsi" / f"{stem}.h5",
            source_path / "train_hsi" / f"{stem}.h5",
        ])

    candidates.extend([
        image_path.parent.parent / f"{image_path.parent.parent.name}_reconstructed_npy" / f"{stem}.npy",
        image_path.parent.parent / "truck_hsi" / f"{stem}.h5",
        image_path.parent.parent / "train_hsi" / f"{stem}.h5",
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_hsi_numpy(path):
    if path is None:
        return None

    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            if "reconstructed_hsi" in f:
                arr = f["reconstructed_hsi"][:].astype(np.float32)
            else:
                first_key = next(iter(f.keys()))
                arr = f[first_key][:].astype(np.float32)
    else:
        raise ValueError(f"Unsupported HSI file: {path}")

    if arr.ndim != 3:
        raise ValueError(f"Expected HSI cube with 3 dims, got {arr.shape} from {path}")

    # Accept BHW, HWB or WHB layouts and convert to BHW. Square spatial dimensions are tested first, so that a
    # low-resolution cube such as 204x16x16 is not taken for a 16-band HWB cube.
    if arr.shape[1] == arr.shape[2] and arr.shape[0] != arr.shape[-1]:
        cube = arr
    elif arr.shape[0] == arr.shape[1] and arr.shape[-1] != arr.shape[0]:
        cube = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] <= 512 and arr.shape[0] < arr.shape[-1]:
        cube = arr
    elif arr.shape[-1] <= 512:
        cube = np.moveaxis(arr, -1, 0)
    else:
        cube = arr

    cube = np.nan_to_num(cube, nan=0.0, posinf=1.0, neginf=0.0)
    return cube.astype(np.float32, copy=False)


def resize_hsi_tensor(cube, resolution):
    if cube is None:
        return None
    tensor = torch.from_numpy(cube).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(resolution[1], resolution[0]), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).clamp(0.0, 1.0)

"""GPU clustering for stage `prepare --device cuda`: the numpy functions of clusters.py written in torch.

Same formulas, same chunking and the same tie rules (first minimum / first maximum). Differences to the CPU reference:
the float32 distance matmul runs in cuBLAS (TF32 off) instead of the CPU BLAS, and the Lloyd sums are accumulated in
float64 instead of numpy's sequential float32 np.add.at; so near-tie labels and the last bits of the centres can
differ. Inputs and outputs are numpy arrays, as in clusters.py.
"""
import numpy as np
import torch

CHUNK = 32768


def describe(device):
    return {'device': 'cuda', 'implementation': 'torch (clusters_gpu.py), float32 cuBLAS distances without TF32, '
                                                'float64 Lloyd sums', 'gpu': torch.cuda.get_device_name(device),
            'torch': torch.__version__}


def _distances(x, centers):
    return ((x * x).sum(1)[:, None] + (centers * centers).sum(1)[None] - 2 * x @ centers.T).clamp_min(0)


def _assign(x, centers):
    labels = torch.empty(len(x), dtype=torch.int64, device=x.device)
    for start in range(0, len(x), CHUNK):
        labels[start:start + CHUNK] = _distances(x[start:start + CHUNK], centers).argmin(1)
    return labels


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _farthest_points(x, materials):
    if materials > len(x):
        raise ValueError('More clusters than observed TRAIN samples')
    remaining = _distances(x, x.mean(0, keepdim=True))[:, 0]
    chosen = []
    for _ in range(materials):
        index = int(remaining.argmax())
        if remaining[index] < 0:
            raise ValueError('Insufficient distinct observed TRAIN support')
        chosen.append(index)
        remaining = torch.minimum(remaining, _distances(x, x[index:index + 1])[:, 0])
        remaining[chosen] = -1
    return chosen


def cluster(x, materials, bands, iterations, device='cuda'):
    """Joint farthest-point seeding, then `iterations` Lloyd updates with the shape part kept on the unit sphere."""
    with _no_tf32():
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        centers = x[_farthest_points(x, materials)].clone()
        for iteration in range(iterations):
            labels = _assign(x, centers)
            counts = torch.bincount(labels, minlength=materials)
            if bool((counts == 0).any()):
                raise ValueError('Empty fixed-K cluster; stop without adaptive K or retry')
            sums = torch.zeros(centers.shape, dtype=torch.float64, device=device).index_add_(0, labels, x.double())
            means = sums / counts[:, None]
            means[:, :bands] = _unit(means[:, :bands])
            centers = means.float()
            print('joint fixed-K iteration', iteration, 'support', counts.tolist(), flush=True)
        return centers.cpu().numpy()


def assign(x, centers, device='cuda'):
    """Nearest-centre labels (int32) of the rows of x."""
    with _no_tf32():
        labels = _assign(torch.from_numpy(np.ascontiguousarray(x)).to(device), torch.from_numpy(centers).to(device))
    return labels.cpu().numpy().astype(np.int32)


class _no_tf32:
    def __enter__(self):
        self.saved = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.saved

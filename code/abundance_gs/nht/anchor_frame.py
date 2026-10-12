"""Anchor Gaussians in the NHT trainer's world frame: ply reading, similarity transform, pixel-footprint dilation."""
import numpy as np
import torch


def load_anchor_ply(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))['vertex']
    means = np.stack([v['x'], v['y'], v['z']], -1).astype(np.float32)
    scales = np.stack([v[f'scale_{i}'] for i in range(3)], -1).astype(np.float32)      # log sigma
    quats = np.stack([v[f'rot_{i}'] for i in range(4)], -1).astype(np.float32)         # w, x, y, z (unnormalised)
    opacities = np.asarray(v['opacity']).astype(np.float32)                            # logit
    dc = np.stack([v[f'f_dc_{i}'] for i in range(3)], -1).astype(np.float32)
    rest_names = sorted([p.name for p in v.properties if p.name.startswith('f_rest_')], key=lambda s: int(s.split('_')[-1]))
    rest = np.stack([v[n] for n in rest_names], -1).astype(np.float32) if rest_names else np.zeros((len(means), 0), np.float32)
    return {'means': means, 'scales': scales, 'quats': quats, 'opacities': opacities, 'dc': dc, 'rest': rest}


def rotmat_to_quat(r):
    """3x3 rotation -> (w, x, y, z)."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    return np.asarray(q, np.float64)


def quat_multiply(a, b):
    """Hamilton product a * b, (w, x, y, z); a is [4], b is [N, 4]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def to_nht_frame(anchor, transform):
    """Apply the parser's similarity transform (x -> s R x + t) to means, scales and rotations."""
    m = np.asarray(transform, np.float64)
    linear = m[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    rot = linear / scale
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5), 'parser transform is not a similarity'
    means = anchor['means'].astype(np.float64) @ linear.T + m[:3, 3]
    quats = anchor['quats'].astype(np.float64)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quat_multiply(rotmat_to_quat(rot), quats)
    out = {'means': torch.from_numpy(means).float(),
           'scales': torch.from_numpy(anchor['scales'] + np.log(scale)).float(),
           'quats': torch.from_numpy(quats).float(),
           'opacities': torch.from_numpy(anchor['opacities']).float(),
           'similarity_scale': scale, 'rotation': rot}
    return out


def pixel_footprint_variance(means, camtoworlds, Ks, width, height, variance_px=0.3):
    """World-space variance equal to ``variance_px`` pixel^2 at each primitive, median over the cameras that see it.

    The anchor's EWA rasteriser adds 0.3 px^2 to every projected covariance (the
    3DGS low-pass filter). The 3D-evaluated NHT kernel has no such term, so the
    transferred primitives can be given the equivalent isotropic 3D variance
    (z / f)^2 * 0.3 instead. Returns [N]; primitives seen by no camera get 0.
    """
    values = torch.full((len(camtoworlds), len(means)), float('nan'), device=means.device)
    for i, (c2w, K) in enumerate(zip(camtoworlds, Ks)):
        w2c = torch.linalg.inv(c2w)
        cam = means @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
        ok = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        values[i, ok] = variance_px * (z[ok] / K[0, 0]) ** 2
    return torch.nan_to_num(torch.nanmedian(values, dim=0).values, nan=0.0)

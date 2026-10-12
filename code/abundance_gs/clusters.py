"""Stage `prepare`: appearance clusters of the multi-view appearance regularization.

The HSI-supervision TRAIN spectra (3x3-smoothed, spectral norm > 0.03) are encoded as unit spectral shape plus a
robust-standardized log brightness and clustered into exactly K appearance clusters (joint farthest-point seeding,
then 20 Lloyd updates with the shape part kept on the unit sphere). Cluster centre k initializes endmember k; the
per-pixel cluster labels y_r are the targets of the persistent assignment regularization. No material GT, TEST
observation or RGB is read.

Cache layout (<cache>/): initial_endmembers.npy [K, B], candidate_clusters.npz, sampling_bins/<view>.npz
(order/counts/starts/prob), dense_labels/<view>.npy, preparation.json (hashes of every input and output),
CACHE_AUDIT.json. An existing cache is never overwritten; it is validated against K, bands, grid, pack cameras and
the TRAIN subset, and every pinned input/output hash, before CUDA is touched. Caches of the older format (other
metadata key names) are read through `read_preparation`.

The clustering (farthest-point seeding, Lloyd updates, per-pixel assignment) runs on the CPU (the reference, numpy) or,
with device='cuda', in torch on the GPU (clusters_gpu.py; float32 distances by cuBLAS instead of the CPU BLAS, so a
few near-tie labels and the last bits of the centres can differ). preparation.json records the device.
"""
import json
from pathlib import Path

import numpy as np

from . import records
from .data import Pack
from .defaults import Clusters

D = Clusters()
SIGNAL_THRESHOLD = D.signal_threshold


sha = records.file_digest                              # SHA256 of a file, streamed


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def smooth(cube):
    """[B, H, W] -> [H, W, B] 3x3 reflect-padded box average (summed band-major, i.e. contiguously; same arithmetic)."""
    _, height, width = cube.shape
    padded = np.pad(cube, ((0, 0), (1, 1), (1, 1)), mode='reflect')
    total = sum(padded[:, dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)) / 9
    return np.ascontiguousarray(total.transpose(1, 2, 0))          # contiguous: numpy's row norms depend on the layout


def smoothed_spectra(pack, views):
    """(view, [H*W, B] smoothed spectra) in view order, loaded ahead in threads."""
    for i, cube in pack.hsi_iter(views):
        yield i, smooth(cube.numpy()).reshape(-1, pack.bands)


def distances(x, centers):
    return np.maximum((x * x).sum(1)[:, None] + (centers * centers).sum(1)[None] - 2 * x @ centers.T, 0)


def assign(x, centers):
    labels = np.empty(len(x), np.int32)
    for start in range(0, len(x), 32768):
        labels[start:start + 32768] = distances(x[start:start + 32768], centers).argmin(1)
    return labels


def farthest_points(x, materials):
    if materials > len(x):
        raise ValueError('More clusters than observed TRAIN samples')
    remaining = distances(x, x.mean(0, keepdims=True))[:, 0]
    chosen = []
    for _ in range(materials):
        index = int(remaining.argmax())
        if remaining[index] < 0:
            raise ValueError('Insufficient distinct observed TRAIN support')
        chosen.append(index)
        remaining = np.minimum(remaining, distances(x, x[index:index + 1])[:, 0])
        remaining[chosen] = -1
    return np.asarray(chosen)


def appearance_descriptors(spectra, median, scale, kappa):
    """Descriptors q_r = [unit spectral shape, kappa * b] (b: robust-standardized log norm, clipped) and the norms."""
    norm = np.linalg.norm(spectra, axis=1)
    relative = np.clip((np.log(np.maximum(norm, 1e-12)) - median) / scale, -4, 4)
    return np.c_[unit(spectra), kappa * relative].astype(np.float32), norm


def cluster(x, materials, bands):
    centers = x[farthest_points(x, materials)].copy()
    for iteration in range(D.iterations):
        labels = assign(x, centers)
        counts = np.bincount(labels, minlength=materials)
        if np.any(counts == 0):
            raise ValueError('Empty fixed-K cluster; stop without adaptive K or retry')
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, x)
        centers = sums / counts[:, None]
        centers[:, :bands] = unit(centers[:, :bands])
        centers = centers.astype(np.float32)
        print('joint fixed-K iteration', iteration, 'support', counts.tolist(), flush=True)
    return centers


def make_bins(labels, materials):
    labels = np.asarray(labels, np.int32)
    pixels = np.flatnonzero(labels >= 0)
    if np.any(labels >= materials):
        raise ValueError('Cluster label exceeds dictionary K')
    order = pixels[np.argsort(labels[pixels], kind='stable')].astype(np.int32)
    counts = np.bincount(labels[pixels], minlength=materials).astype(np.int64)
    starts = np.r_[0, counts.cumsum()[:-1]].astype(np.int64)
    probability = np.sqrt(counts) * (counts >= 16)                 # kept for cache compatibility (unused)
    probability /= probability.sum() if probability.sum() else 1
    return dict(order=order, counts=counts, starts=starts, prob=probability)


# older cache format -> current metadata keys
_OLDER_KEYS = {'full_train_teacher_counts': 'full_train_label_counts', 'teacher_semantics': 'label_semantics'}


def read_preparation(cache):
    """preparation.json of a cache, with keys of the older cache format mapped to the current names."""
    metadata = json.loads((Path(cache) / 'preparation.json').read_text())
    return {_OLDER_KEYS.get(k, k): v for k, v in metadata.items()}


def output_digest(metadata, out, path):
    """Pinned hash of a cache output; relocated caches are matched by their relative path."""
    manifest = metadata.get('output_sha256', {})
    path, out = Path(path), Path(out)
    if str(path) in manifest:
        return manifest[str(path)]
    relative = path.relative_to(out).as_posix()
    candidates = [value for name, value in manifest.items() if name == relative or name.endswith('/' + relative)]
    if len(candidates) != 1:
        raise ValueError('No unique pinned cache output: ' + str(path))
    return candidates[0]


def read_bins(path, materials, pixels):
    """Label bins of one view and the dense label vector they encode (-1 = not assigned)."""
    with np.load(path, allow_pickle=False) as arrays:
        bins = {key: arrays[key].copy() for key in arrays.files}
    if not {'order', 'counts', 'starts', 'prob'}.issubset(bins):
        raise ValueError('Label bins require order/counts/starts/prob')
    for key in ('order', 'counts', 'starts'):
        if bins[key].ndim != 1 or not np.issubdtype(bins[key].dtype, np.integer):
            raise ValueError('Label bin indices must be one-dimensional integers')
    counts, starts, order = bins['counts'], bins['starts'], bins['order']
    if len(counts) != materials or len(starts) != materials or np.any(counts < 0):
        raise ValueError('Label bin counts must contain exactly K nonnegative entries')
    if not np.array_equal(starts, np.r_[0, counts.cumsum()[:-1]]):
        raise ValueError('Label bin offsets do not match K counts')
    if len(order) != int(counts.sum()) or len(np.unique(order)) != len(order):
        raise ValueError('Duplicate or missing signal-supported TRAIN pixels')
    if np.any(order < 0) or np.any(order >= pixels):
        raise ValueError('Label pixel index is outside the training grid')
    probability = bins['prob']
    if probability.shape != (materials,) or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError('Invalid sampling probabilities')
    if len(order) and not np.isclose(probability.sum(), 1., atol=1e-6) and np.any(counts >= 16):
        raise ValueError('Sampling probabilities must sum to one')
    labels = np.full(pixels, -1, np.int64)
    for k, count in enumerate(counts):
        labels[order[starts[k]:starts[k] + count]] = k
    return bins, labels


def validate_cache(out, expected_train=None, expected_materials=None, verify_sources=True, pack=None):
    """Check a fixed-K cache entirely on the CPU; returns the audit record."""
    out = Path(out)
    metadata = read_preparation(out)
    if not metadata.get('complete') or metadata.get('GT_or_TEST_read_for_initialization') is not False:
        raise ValueError('Incomplete or non-TRAIN-only preparation')
    if metadata.get('signal_threshold', SIGNAL_THRESHOLD) != SIGNAL_THRESHOLD:
        raise ValueError(f'Standard appearance preparation uses the fixed {SIGNAL_THRESHOLD} signal gate')
    materials = metadata['materials']
    if expected_materials is not None and materials != expected_materials:
        raise ValueError('Cached K differs from requested dictionary capacity')
    if expected_train is not None and metadata['train_views'] != expected_train:
        raise ValueError('Cached TRAIN views differ from actual HSI supervision subset')
    initial_path = out / 'initial_endmembers.npy'
    if sha(initial_path) != output_digest(metadata, out, initial_path):
        raise ValueError('Initial dictionary checksum differs')
    initial = np.load(initial_path, allow_pickle=False)
    bands = metadata.get('bands', pack.bands if pack is not None else initial.shape[-1])
    resolution = metadata.get('resolution_wh', [pack.width, pack.height] if pack is not None else None)
    if resolution is None:
        raise ValueError('A cache without resolution_wh needs its pack to verify pixel coverage')
    if pack is not None and (bands != pack.bands or resolution != [pack.width, pack.height]):
        raise ValueError('Cached bands or training grid differ from actual pack')
    pixels = int(np.prod(resolution))
    if initial.shape != (materials, bands) or not np.isfinite(initial).all() or np.any(initial < 0):
        raise ValueError('Invalid K-by-band cluster centre array')
    if not np.allclose(np.linalg.norm(initial, axis=1), 1., atol=1e-6):
        raise ValueError('Cluster centre spectral directions must be unit normalized')
    candidate = out / 'candidate_clusters.npz'
    if not candidate.is_file() or sha(candidate) != output_digest(metadata, out, candidate):
        raise ValueError('Pinned joint candidate cluster file is missing or changed')
    with np.load(candidate, allow_pickle=False) as arrays:
        if arrays['centers'].shape != initial.shape or not np.array_equal(initial, arrays['centers']):
            raise ValueError('Cache centers and initial E have different endmember ordering')
        if 'joint_centers' not in arrays:
            raise ValueError('Standard cache requires joint spectral-shape/brightness coordinates')
        joint = arrays['joint_centers']
        if joint.shape != (materials, bands + 1) or not np.isfinite(joint).all() or not np.array_equal(joint[:, :bands], initial):
            raise ValueError('Joint center coordinates and initial E differ')
    source_checks = []
    if verify_sources and pack is not None:
        manifest = metadata.get('source_sha256', {})
        pack_path = pack.directory / 'pack.json'
        expected = manifest.get(str(pack_path))
        if expected is None:
            matches = [digest for name, digest in manifest.items() if name.endswith('/pack.json')]
            if len(matches) != 1:
                raise ValueError('No unique preparation pack checksum')
            expected = matches[0]
        if sha(pack_path) != expected:
            raise ValueError('Actual pack cameras/normalization/splits changed since preparation')
        train_paths = [pack.views[i]['hsi_path'] for i in pack.train]
        forbidden_paths = {v['hsi_path'] for i, v in enumerate(pack.views) if i not in set(pack.train)}
        if forbidden_paths.intersection(manifest) - set(train_paths):
            raise ValueError('Preparation recorded an observation outside actual HSI supervision')
        for path in train_paths:
            digest = manifest.get(path)
            if digest is None or sha(path) != digest:
                raise ValueError('Actual TRAIN observation is unpinned or changed: ' + path)
            source_checks.append(path)
    totals = np.zeros(materials, np.int64)
    per_view = []
    for name in metadata['train_views']:
        bin_path = out / 'sampling_bins' / (name + '.npz')
        if sha(bin_path) != output_digest(metadata, out, bin_path):
            raise ValueError('Label bin checksum differs: ' + name)
        bins, labels = read_bins(bin_path, materials, pixels)
        dense_path = out / 'dense_labels' / (name + '.npy')
        if dense_path.is_file():
            if sha(dense_path) != output_digest(metadata, out, dense_path):
                raise ValueError('Dense label checksum differs: ' + name)
            dense = np.load(dense_path, allow_pickle=False)
            if (not np.issubdtype(dense.dtype, np.integer) or dense.shape != (resolution[1], resolution[0])
                    or not np.array_equal(dense.reshape(-1), labels)):
                raise ValueError('Dense labels and trainer NLL targets differ')
        totals += bins['counts']
        per_view.append(dict(view=name, pixels=pixels, eligible=int(len(bins['order'])),
                             excluded=pixels - len(bins['order']), counts=bins['counts'].tolist()))
    eligible = int(totals.sum())
    total_pixels = pixels * len(metadata['train_views'])
    if 'full_train_label_counts' in metadata and not np.array_equal(totals, metadata['full_train_label_counts']):
        raise ValueError('TRAIN label population metadata differs')
    if 'eligible_train_pixels' in metadata and eligible != metadata['eligible_train_pixels']:
        raise ValueError('Eligible pixel metadata differs')
    if np.any(totals == 0):
        raise ValueError('Cluster has no TRAIN support')
    return dict(complete=True, materials=materials, bands=bands, TRAIN_views=len(metadata['train_views']),
                total_train_pixels=total_pixels, eligible_train_pixels=eligible,
                excluded_signal_pixels=total_pixels - eligible, signal_support_fraction=eligible / total_pixels,
                signal_threshold=metadata.get('signal_threshold', SIGNAL_THRESHOLD), full_train_label_counts=totals.tolist(),
                all_supported_pixels_assigned_exactly_once=True,
                initial_E_and_NLL_bins_have_identical_slot_order=True,
                actual_TRAIN_input_hashes_verified=bool(source_checks),
                TRAIN_only=True, GT_or_TEST_used=False, per_view=per_view,
                preparation_sha256=sha(out / 'preparation.json'))


def ensure_cache(pack, out, materials, device='cpu'):
    """Reuse an exactly matching valid cache; build an absent one; never overwrite."""
    out = Path(out).resolve()
    if materials <= 0:
        raise ValueError('Requested K must be positive')
    if not out.exists():
        prepare(pack.directory, out, materials, train_indices=pack.train, device=device)
    elif not out.is_dir():
        raise ValueError('Cluster cache path is not a directory')
    audit = validate_cache(out, [pack.names[i] for i in pack.train], materials, pack=pack)
    return out, read_preparation(out), audit


class Clustering:
    """Seeding, Lloyd updates and nearest-centre assignment on the CPU (numpy reference) or the GPU (torch)."""

    def __init__(self, device='cpu'):
        self.device = device
        if device != 'cpu':
            from . import clusters_gpu
            self.gpu = clusters_gpu
            self.record = clusters_gpu.describe(device)
        else:
            self.record = {'device': 'cpu', 'implementation': 'numpy (reference)'}

    def cluster(self, x, materials, bands):
        return cluster(x, materials, bands) if self.device == 'cpu' else self.gpu.cluster(x, materials, bands, D.iterations, self.device)

    def assign(self, x, centers):
        return assign(x, centers) if self.device == 'cpu' else self.gpu.assign(x, centers, self.device)


def sample_train_spectra(pack, rng):
    """D.samples_per_view random signal-supported smoothed spectra per TRAIN view; input hashes and signal statistics."""
    samples, view_indices, hashes, signal = [], [], {}, []
    for i, spectra in smoothed_spectra(pack, pack.train):
        norms = np.linalg.norm(spectra, axis=1)
        eligible = np.flatnonzero(norms > SIGNAL_THRESHOLD)
        indices = rng.choice(eligible, min(D.samples_per_view, len(eligible)), replace=False)
        samples.append(spectra[indices])
        view_indices.extend([i] * len(indices))
        path = pack.views[i]['hsi_path']
        hashes[path] = sha(path)
        signal.append(dict(view=pack.names[i], pixels=len(norms), eligible=len(eligible),
                           norm_quantiles=np.quantile(norms, [0, .01, .05, .5, .95, 1]).tolist()))
    return np.concatenate(samples).astype(np.float32), np.asarray(view_indices), hashes, signal


def encoding_statistics(observed):
    """Median and robust scale of the log brightness, and the brightness weight kappa of the descriptors q_r."""
    flux = np.log(np.linalg.norm(observed, axis=1))
    median = float(np.median(flux))
    scale = float(max(.1, 1.4826 * np.median(np.abs(flux - median))))
    directions = unit(observed)
    kappa = float(np.clip(.5 * np.median(np.linalg.norm(directions - unit(directions.mean(0)), axis=1)), .05, .2))
    return median, scale, kappa


def label_train_views(pack, out, clustering, centers, encoding, materials):
    """Pseudo-labels y_r (dense maps and label bins) of every TRAIN pixel (-1 below the signal gate); returns the
    per-cluster population and the files written."""
    bins, dense = out / 'sampling_bins', out / 'dense_labels'
    bins.mkdir()
    dense.mkdir()
    population = np.zeros(materials, np.int64)
    for i, spectra in smoothed_spectra(pack, pack.train):
        features, norm = appearance_descriptors(spectra, *encoding)
        labels = clustering.assign(features, centers)
        labels[norm <= SIGNAL_THRESHOLD] = -1
        name = pack.names[i]
        np.save(dense / (name + '.npy'), labels.reshape(pack.height, pack.width))
        b = make_bins(labels, materials)
        population += b['counts']
        np.savez_compressed(bins / (name + '.npz'), **b)
    return population, [*sorted(bins.glob('*.npz')), *sorted(dense.glob('*.npy'))]


def prepare(pack_path, out, materials, train_indices=None, device='cpu'):
    import torch
    torch.set_num_threads(2)
    out = Path(out)
    if out.exists():
        raise FileExistsError('No overwrite or implicit preparation retry: ' + str(out))
    out.mkdir(parents=True)
    pack = Pack(pack_path)
    if train_indices is not None:
        pack.train = list(train_indices)
    if not pack.train or any(pack.views[i]['split'] != 'train' for i in pack.train):
        raise ValueError('Initialization requires nonempty actual HSI-supervision TRAIN views')
    clustering = Clustering(device)
    observed, view_indices, hashes, signal = sample_train_spectra(pack, np.random.default_rng(D.seed))
    if len(observed) < materials:
        raise ValueError('Insufficient signal-supported TRAIN spectra for requested K; choose K explicitly')
    median, scale, kappa = encoding_statistics(observed)
    features, _ = appearance_descriptors(observed, median, scale, kappa)
    centers = clustering.cluster(features, materials, pack.bands)
    sampled_labels = clustering.assign(features, centers)
    sampled_counts = np.bincount(sampled_labels, minlength=materials)
    sampled_view_support = [len(np.unique(view_indices[sampled_labels == k])) for k in range(materials)]
    np.save(out / 'initial_endmembers.npy', centers[:, :pack.bands])
    np.savez_compressed(out / 'candidate_clusters.npz', centers=centers[:, :pack.bands], joint_centers=centers,
                        support=sampled_counts, selected=np.arange(materials), sampled_view_support=sampled_view_support)
    population, labelled = label_train_views(pack, out, clustering, centers, (median, scale, kappa), materials)
    outputs = [out / 'initial_endmembers.npy', out / 'candidate_clusters.npz', *labelled]
    here = Path(__file__).resolve()
    record = dict(complete=True, algorithm='fixed-K joint unit spectrum + robust-standardized log brightness; joint FPS then20 spherical-shape constrained Lloyd updates; exactly K aligned label/dictionary slots',
                  materials=materials, bands=pack.bands, fixed_K_not_selected_from_GT_or_scores=True,
                  train_views=[pack.names[i] for i in pack.train], resolution_wh=[pack.width, pack.height],
                  seed=D.seed, samples_per_view=D.samples_per_view, sampled_spectra=len(observed),
                  kmeans_iterations=D.iterations, overcomplete_or_auto_K=False,
                  log_brightness_median=median, log_brightness_robust_scale=scale, brightness_feature_weight=kappa,
                  signal_threshold=SIGNAL_THRESHOLD, local_smoothing='3x3 reflect-padded spatial box average',
                  sampled_cluster_counts=sampled_counts.tolist(), sampled_cluster_view_support=sampled_view_support,
                  full_train_label_counts=population.tolist(), eligible_train_pixels=int(population.sum()),
                  excluded_signal_pixels_label=-1, signal_support=signal,
                  label_semantics='observed appearance clusters, not physical materials; low-signal exclusions are not semantic masks',
                  GT_or_TEST_read_for_initialization=False, RGB_read_for_initialization=False,
                  source_sha256={str(here): sha(here), str(here.parent / 'data.py'): sha(here.parent / 'data.py'),
                                 str(Path(pack_path) / 'pack.json'): sha(Path(pack_path) / 'pack.json'),
                                 **hashes},
                  output_sha256={str(path): sha(path) for path in outputs})
    record['clustering'] = clustering.record
    if device != 'cpu':
        gpu_file = here.parent / 'clusters_gpu.py'
        record['source_sha256'][str(gpu_file)] = sha(gpu_file)
    write(out / 'preparation.json', record)
    audit = validate_cache(out, record['train_views'], materials, pack=pack)
    write(out / 'CACHE_AUDIT.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


def load_pseudo_labels(cache, pack, materials):
    """Initial dictionary [K, B] and per-TRAIN-view pseudo-labels y_r [H*W] (-1 = unassigned), hash-checked."""
    import torch
    cache = Path(cache)
    preparation = read_preparation(cache)
    assert preparation['complete'] and preparation['materials'] == materials
    assert preparation['train_views'] == [pack.names[i] for i in pack.train]
    initial_file = cache / 'initial_endmembers.npy'
    assert sha(initial_file) == output_digest(preparation, cache, initial_file)
    initial = torch.from_numpy(np.load(initial_file)).float()
    assert initial.shape == (materials, pack.bands) and torch.isfinite(initial).all()
    labels = {}
    for i in pack.train:
        f = cache / 'sampling_bins' / (pack.names[i] + '.npz')
        assert sha(f) == output_digest(preparation, cache, f)
        labels[i] = torch.from_numpy(read_bins(f, materials, pack.pixels)[1])
    return initial, labels, preparation


def farthest_point_endmembers(pack, materials, pixels_per_view=2048, seed=42):
    """Reference-preset initialization: TRAIN-only angular farthest-point sampling of unit spectra (no cluster cache)."""
    import torch
    rng = np.random.default_rng(seed)
    samples = []
    for i in pack.train:
        x = pack.hsi(i).numpy().astype(np.float32).clip(0, 1)
        x = x.transpose(1, 2, 0).reshape(-1, x.shape[0])
        norm = np.linalg.norm(x, axis=1)
        valid = np.flatnonzero(norm > SIGNAL_THRESHOLD)
        ids = rng.choice(valid, min(len(valid), pixels_per_view), replace=False)
        samples.append(x[ids] / norm[ids, None])
    x = np.concatenate(samples)
    center = x.mean(0)
    center /= np.linalg.norm(center)
    dist = 1 - x @ center
    chosen = []
    for k in range(materials):
        j = int(dist.argmax())
        chosen.append(j)
        new = 1 - x @ x[j]
        dist = new if k == 0 else np.minimum(dist, new)
        dist[chosen] = -1
    return torch.from_numpy(x[chosen].copy()).float()


def run(args):
    if args.materials <= 0:
        raise ValueError('--materials must be positive')
    pack = Pack(args.pack)
    pack.limit_hsi_training(args.train_max_views, args.val_every_n_train)
    if args.check_only:
        audit = validate_cache(args.output, [pack.names[i] for i in pack.train], args.materials, pack=pack)
    else:
        created = not Path(args.output).exists()
        inputs, config = records.expected('prepare', args)
        _, _, audit = ensure_cache(pack, args.output, args.materials, device=args.device)
        if created:                                    # existing caches are never written to
            records.complete(args.output, 'prepare', inputs, config, cache=str(Path(args.output).resolve()))
    print(json.dumps(dict(cache=str(Path(args.output).resolve()), audit=audit), indent=2))

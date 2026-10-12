#!/usr/bin/env python3
"""Common TRAIN-calibrated NVIDIA material scoring, without dataset mutations.

Consumes predictions/<stem>_abundance.npz (CHW abundance and alpha) and
endmembers_radiance.npy (K x 64, actual radiance units). Visibility-normalized
latent shares are target-agreement diagnostics, not identified physical area.
The released surface_evaluation_mask is used identically for all methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from abundance_gs import paths
from abundance_gs.scorers import benchmark as ebm

CACHE = paths.BUILD_DIR / 'scorer_cache' / 'nvidia_reference'   # pure-TRAIN reference spectra cache

SPACE, CLASSES, BANDS = 'classes8', 8, 64
MASK_NOTE = 'Released surface_evaluation_mask, exactly coverage >= 0.5; material labels outside this mask are set to -1 in evaluator memory only. Sky-dominated footprints excluded.'
ERROR_NOTE = ('L1-normalized exported latent coefficients, class-summed under the frozen TRAIN map. '
              'Unassigned mass is retained, never redistributed. Area is conditional first-surface coverage; '
              'radiance is first-surface-attributed outgoing power over 400--1000 nm, including glass-reflected paths '
              'and excluding first-leg atmospheric haze/background. Neither target is irradiance; '
              'agreement does not identify intrinsic reflectance or physical occupancy.')


def linear_sum_assignment(cost):
    """SciPy's linear_sum_assignment."""
    from scipy.optimize import linear_sum_assignment as solve
    return solve(cost)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)


def split_views(dataset):
    splits = {split: ebm.stems_for_split(dataset, split) for split in ('train', 'val', 'test')}
    if [len(splits[s]) for s in ('train', 'val', 'test')] != [180, 28, 32]:
        raise ValueError('Expected the complete NVIDIA 180 TRAIN / 28 VAL / 32 TEST release')
    all_views = sum(splits.values(), [])
    if len(set(all_views)) != 240:
        raise ValueError('Duplicate or overlapping camera split')
    return splits


def read_gt(dataset, stem, targets=False):
    with np.load(dataset / 'gt' / f'{stem}.npz') as z:
        labels = z['material_merged'].astype(np.int64)
        coverage = z['coverage'].astype(np.float32)
        support = z['surface_evaluation_mask'].astype(bool)
        if not np.array_equal(support, coverage >= .5):
            raise ValueError(f'{stem}: explicit surface mask and coverage gate disagree')
        if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
            raise ValueError(f'{stem}: invalid coverage')
        if labels[support].size and (labels[support].min() < 0 or labels[support].max() >= CLASSES):
            raise ValueError(f'{stem}: material outside the frozen eight-class taxonomy')
        # First-surface labels can be present in a sky-dominated mixed footprint.
        # The coverage protocol, not the presence of an ID, defines evaluation.
        labels = np.where(support, labels, -1)
        truth = {}
        if targets:
            for target, key in [('area', 'abundance_area_merged'),
                                ('radiance', 'abundance_radiance_weighted_class_merged')]:
                array = np.moveaxis(z[key], -1, 0).astype(np.float32)
                if array.shape != (CLASSES, *labels.shape) or not np.isfinite(array).all() or array.min() < -1e-7:
                    raise ValueError(f'{stem}: invalid {target} truth')
                if support.any() and not np.allclose(array[:, support].sum(0), 1, atol=2e-5, rtol=0):
                    raise ValueError(f'{stem}: {target} truth is not conditional normalized class fraction')
                truth[target] = array
    return labels, support, coverage, truth


def read_prediction(directory, stem, channels=None):
    path = directory / f'{stem}_abundance.npz'
    with np.load(path) as z:
        abundance, active = ebm.normalize_abundance(z['abundance'], 'chw')
        alpha = np.asarray(z['alpha'], np.float32)
    if alpha.shape != abundance.shape[1:] or not np.isfinite(alpha).all():
        raise ValueError(f'{path}: finite alpha must match native prediction footprint')
    if channels is not None and len(abundance) != channels:
        raise ValueError(f'{path}: learned channel count changed')
    # Alpha is recorded but does not select scoring support or calibration votes.
    return abundance, active


def calibrate(dataset, predictions, train, names):
    votes = mass = None
    sources = []
    support_pixels = 0
    for stem in train:
        abundance, _ = read_prediction(predictions, stem, None if votes is None else len(votes))
        labels, support, _, _ = read_gt(dataset, stem)
        if abundance.shape[1:] != labels.shape:
            raise ValueError(f'{stem}: no implicit prediction/GT resizing is permitted')
        if votes is None:
            votes = np.zeros((len(abundance), CLASSES), np.int64)
            mass = np.zeros(len(abundance), np.float64)
        mass += abundance[:, support].sum(1, dtype=np.float64)
        ebm.add_votes(votes, abundance, labels)
        support_pixels += int(support.sum())
        sources.append(dict(view=stem, prediction_sha256=sha(predictions / f'{stem}_abundance.npz')))
    mapping = ebm.mapping_from_votes(votes)
    payload = dict(algorithm='global many-to-one dominant-channel majority vote; upstream add_votes/mapping_from_votes',
                   calibration_split='train', calibration_views=train, held_out_gt_used_for_mapping=False,
                   foreground_mask=MASK_NOTE, K=len(mapping), class_names=names,
                   mapping_tie_rule='lowest class ID among tied positive votes',
                   unassigned_channel_rule='No dominant supported TRAIN votes -> -1; no invented class.',
                   support_pixels=support_pixels,
                   channel_foreground_training_abundance_mass=mass.tolist(),
                   prediction_sources=sources,
                   label_spaces={SPACE: dict(channel_to_class=mapping.tolist(), votes=votes.tolist(),
                                            unassigned_channels=np.flatnonzero(mapping < 0).tolist(),
                                            nonempty_but_never_dominant_channels=np.flatnonzero((mass > 1e-12) & (mapping < 0)).tolist(),
                                            fitted_on='all supported TRAIN pixels only')})
    payload['sha256'] = digest_json(payload)
    return mapping, payload


def errors_payload(states):
    result = {}
    for target, state in states.items():
        n = state['pixels']
        if not n:
            raise ValueError('No supported pixels for composition evaluation')
        result[target] = {'all': dict(pixels=n, RMSE=float(np.sqrt(state['squared'].sum() / (n * CLASSES))),
                                     MAE=float(state['absolute'].sum() / (n * CLASSES)),
                                     RMSE_per_class=np.sqrt(state['squared'] / n).tolist(),
                                     MAE_per_class=(state['absolute'] / n).tolist())}
    return result


def evaluate(dataset, predictions, mapping, held, names):
    dominant_cm = np.zeros((CLASSES, CLASSES + 1), np.int64)
    summed_cm = np.zeros_like(dominant_cm)
    states = {target: dict(squared=np.zeros(CLASSES), absolute=np.zeros(CLASSES), pixels=0)
              for target in ('area', 'radiance')}
    per_view, sources, pixels, unknown_mass, inactive_pixels = [], [], 0, 0., 0
    for stem in held:
        abundance, active = read_prediction(predictions, stem, len(mapping))
        labels, support, _, truth = read_gt(dataset, stem, targets=True)
        if abundance.shape[1:] != labels.shape:
            raise ValueError(f'{stem}: prediction/GT footprint mismatch')
        class_abundance, unknown = ebm.aggregate_channels(abundance, mapping, CLASSES)
        dominant, summed = ebm.mapped_labels(abundance, mapping, class_abundance, unknown)
        dcm, scm = ebm.confusion(labels, dominant, CLASSES), ebm.confusion(labels, summed, CLASSES)
        dominant_cm += dcm
        summed_cm += scm
        count = int(support.sum())
        pixels += count
        unknown_mass += float(unknown[support].sum(dtype=np.float64))
        inactive_pixels += int((support & ~active).sum())
        view_states = {}
        for target, value in truth.items():
            delta = (class_abundance[:, support] - value[:, support]).astype(np.float64)
            squared, absolute = (delta * delta).sum(1), np.abs(delta).sum(1)
            states[target]['squared'] += squared
            states[target]['absolute'] += absolute
            states[target]['pixels'] += count
            view_states[target] = dict(squared=squared, absolute=absolute, pixels=count)
        per_view.append(dict(view=stem, support_pixels=count,
                             mapped_dominant_slot_material_agreement=ebm.confusion_metrics(dcm, names),
                             summed_class_abundance_material_agreement=ebm.confusion_metrics(scm, names),
                             class_abundance_errors=errors_payload(view_states),
                             mean_unassigned_foreground_abundance_mass=float(unknown[support].mean()),
                             inactive_supported_prediction_pixels=int((support & ~active).sum())))
        sources.append(dict(view=stem, prediction_sha256=sha(predictions / f'{stem}_abundance.npz')))
    result = dict(mapped_dominant_slot_material_agreement=ebm.confusion_metrics(dominant_cm, names),
                  summed_class_abundance_material_agreement=ebm.confusion_metrics(summed_cm, names),
                  mean_unassigned_foreground_abundance_mass=unknown_mass / pixels,
                  inactive_supported_prediction_pixels=inactive_pixels,
                  class_abundance_errors=errors_payload(states), class_abundance_error_note=ERROR_NOTE)
    return dict(split='test', views=held, mapping_refitted_on_scored_views=False,
                foreground_mask=MASK_NOTE, label_spaces={SPACE: result}, per_view=per_view,
                prediction_sources=sources, abundance_statistic_scope=ERROR_NOTE)


def reference_identity(dataset, clean, train):
    return dict(dataset=str(dataset.resolve()), clean_dataset=str(clean.resolve()), train_views=train,
                material_protocol_sha256=sha(dataset / 'material_class_protocol.json'),
                transform_train_sha256=sha(dataset / 'transforms_train.json'),
                pure_rule='abundance_area_merged[...,c] * coverage >= 0.9 and released surface_evaluation_mask',
                reference_space='actual stored clean radiance, all qualifying native TRAIN pixels, equally weighted',
                inputs=[dict(view=stem, clean=identity(clean / 'hsi' / f'{stem}.npy'),
                             ground_truth=identity(dataset / 'gt' / f'{stem}.npz')) for stem in train])


def training_references(dataset, clean, train, names, cache_directory):
    """Reusable CPU cache; no held-out cube or ground truth enters its construction."""
    source = reference_identity(dataset, clean, train)
    key = digest_json(source)
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / f'pure_train_radiance_{key}.npz'
    if path.exists():
        with np.load(path) as z:
            references, counts = z['references'].copy(), z['pixel_counts'].copy()
            recorded = json.loads(str(z['source_json']))
        if recorded != source or references.shape != (CLASSES, BANDS) or np.any(counts < 0):
            raise ValueError('Invalid or stale pure-TRAIN reference cache')
    else:
        sums, counts = np.zeros((CLASSES, BANDS), np.float64), np.zeros(CLASSES, np.int64)
        for index, stem in enumerate(train):
            _, support, coverage, truth = read_gt(dataset, stem, targets=True)
            cube = np.load(clean / 'hsi' / f'{stem}.npy', mmap_mode='r')
            if cube.shape != (*support.shape, BANDS):
                raise ValueError(f'{stem}: clean radiance must be native HWC64')
            for cls in range(CLASSES):
                pure = support & (truth['area'][cls] * coverage >= .9)
                n = int(pure.sum())
                if n:
                    selected = np.asarray(cube[pure])
                    if not np.isfinite(selected).all() or selected.min() < -1e-7:
                        raise ValueError(f'{stem}: invalid clean radiance')
                    sums[cls] += selected.sum(0, dtype=np.float64)
                    counts[cls] += n
            if (index + 1) % 30 == 0:
                print(f'Pure TRAIN references: {index + 1}/{len(train)}', flush=True)
        references = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
        if reference_identity(dataset, clean, train) != source:
            raise ValueError('Reference source changed during accumulation')
        temporary = path.with_name(f'.{path.stem}.{os.getpid()}.npz')
        np.savez_compressed(temporary, references=references, pixel_counts=counts,
                            source_json=np.asarray(json.dumps(source, sort_keys=True)))
        temporary.replace(path)
    receipt = dict(reference_cache=str(path), reference_cache_sha256=sha(path), source_identity_sha256=key,
                   calibration_split='train', reference_views=train, held_out_reference_used=False,
                   class_names=names, pure_pixel_counts=counts.tolist(),
                   pure_rule=source['pure_rule'], reference_space=source['reference_space'])
    return references, counts, receipt


def endmember_error(endmembers, references, counts, names, receipt):
    learned = np.load(endmembers).astype(np.float64)
    if learned.ndim != 2 or learned.shape[1] != BANDS or not np.isfinite(learned).all():
        raise ValueError('Learned endmembers must be finite K x 64 radiance spectra')
    valid = np.flatnonzero((counts > 0) & (np.linalg.norm(references, axis=1) > 1e-12))
    if np.any(np.linalg.norm(learned, axis=1) <= 1e-12):
        raise ValueError('Zero learned spectrum has undefined spectral angle')
    if len(learned) < len(valid):
        raise ValueError('One-to-one EM requires at least one learned slot per supported reference class')
    angles = ebm.sam_matrix(learned, references[valid])
    channels, columns = linear_sum_assignment(angles)
    matched = [dict(channel=int(k), class_id=int(valid[c]), class_name=names[int(valid[c])],
                    spectral_angle_degrees=float(angles[k, c])) for k, c in zip(channels, columns)]
    return dict(mean_SAM_degrees=float(np.mean([row['spectral_angle_degrees'] for row in matched])) if matched else None,
                matching='One global one-to-one Hungarian minimum total spectral angle, independent of material vote mapping.',
                reference=receipt, endmembers_radiance_file=str(endmembers), endmembers_sha256=sha(endmembers),
                valid_reference_classes=valid.tolist(), omitted_no_pure_train_reference_classes=np.setdiff1d(np.arange(CLASSES), valid).tolist(),
                per_match=matched, interpretation='Agreement with observed pure-TRAIN mean radiance, not identified reflectance.')


def self_test():
    # Distinguishes dominant slot from a material split across two slots.
    abundance = np.array([.3, .3, .4, 0.], np.float32)[:, None, None]
    mapping = np.array([0, 0, 1, -1])
    grouped, unknown = ebm.aggregate_channels(abundance, mapping, CLASSES)
    dominant, summed = ebm.mapped_labels(abundance, mapping, grouped, unknown)
    assert dominant.item() == 1 and summed.item() == 0
    abundance[:, 0, 0] = [0., .6, 0., .4]
    grouped, unknown = ebm.aggregate_channels(abundance, mapping, CLASSES)
    assert np.isclose(grouped[0].item(), .6) and np.isclose(unknown.item(), .4)
    votes = np.array([[3, 3] + [0] * 6, [0] * 8], np.int64)
    assert ebm.mapping_from_votes(votes).tolist() == [0, -1]
    # Coverage below .5 cannot cast a TRAIN vote even when a raw channel dominates.
    with tempfile.TemporaryDirectory(prefix='nvidia-score-test-') as temporary:
        ds = Path(temporary)
        (ds / 'gt').mkdir()
        labels = np.array([[7, 2, 3]])
        coverage = np.array([[.499, .5, 1.]], np.float32)
        np.savez(ds / 'gt/f_0000.npz', material_merged=labels, coverage=coverage,
                 surface_evaluation_mask=coverage >= .5)
        actual, _, _, _ = read_gt(ds, 'f_0000')
        ab = np.ones((1, 1, 3), np.float32)
        vote = np.zeros((1, CLASSES), np.int64)
        ebm.add_votes(vote, ab, actual)
        assert vote.sum() == 2 and vote[0, 2] == 1 and vote[0, 3] == 1
        # A mismatch between recorded mask and coverage must fail.
        np.savez(ds / 'gt/f_0000.npz', material_merged=labels, coverage=coverage,
                 surface_evaluation_mask=np.ones_like(coverage, bool))
        try:
            read_gt(ds, 'f_0000')
        except ValueError:
            pass
        else:
            raise AssertionError('Coverage mismatch was accepted')
        # Integrate NPZ input, TRAIN-only calibration, support and TEST errors.
        pred = ds / 'predictions'
        pred.mkdir()
        class_names = [str(i) for i in range(CLASSES)]
        labels = np.array([[7, 0, 1]])
        coverage = np.array([[.499, 1., 1.]], np.float32)
        truth = np.eye(CLASSES, dtype=np.float32)[labels]
        for view, dominant_slot in [('f_0001', 0), ('f_0002', 1)]:
            np.savez(ds / 'gt' / f'{view}.npz', material_merged=labels, coverage=coverage,
                     surface_evaluation_mask=coverage >= .5)
            a = np.zeros((4, 1, 3), np.float32)
            a[3, 0, 0], a[dominant_slot, 0, 1], a[2, 0, 2] = 1, 1, 1
            np.savez(pred / f'{view}_abundance.npz', abundance=a.astype(np.float16), alpha=np.zeros((1, 3)))
        calibrated, _ = calibrate(ds, pred, ['f_0001', 'f_0002'], class_names)
        assert calibrated.tolist() == [0, 0, 1, -1]  # sky-dominated slot does not gain class7.
        a = np.zeros((4, 1, 3), np.float32)
        a[:, 0, 1], a[2, 0, 2] = [.3, .3, .4, 0.], 1
        np.savez(pred / 'f_0003_abundance.npz', abundance=a.astype(np.float16), alpha=np.zeros((1, 3)))
        np.savez(ds / 'gt/f_0003.npz', material_merged=labels, coverage=coverage,
                 surface_evaluation_mask=coverage >= .5,
                 abundance_area_merged=truth, abundance_radiance_weighted_class_merged=truth)
        evaluated = evaluate(ds, pred, calibrated, ['f_0003'], class_names)['label_spaces'][SPACE]
        assert evaluated['mapped_dominant_slot_material_agreement']['foreground_miou'] == .25
        assert evaluated['summed_class_abundance_material_agreement']['foreground_miou'] == 1
        assert evaluated['class_abundance_errors']['area']['all']['pixels'] == 2
        assert calibrated.tolist() == [0, 0, 1, -1]  # TEST evaluation cannot refit this map.
        # EM recovers the permutation from radiance direction, independently of vote labels.
        endmembers = ds / 'endmembers.npy'
        reference = np.eye(BANDS)[:CLASSES]
        np.save(endmembers, reference[::-1] * 3)
        em = endmember_error(endmembers, reference, np.ones(CLASSES, np.int64), [str(i) for i in range(CLASSES)], {})
        assert em['mean_SAM_degrees'] == 0 and [r['class_id'] for r in em['per_match']] == list(range(7, -1, -1))
    zero = np.zeros_like(abundance)
    grouped, unknown = ebm.aggregate_channels(zero, mapping, CLASSES)
    dominant, summed = ebm.mapped_labels(zero, mapping, grouped, unknown)
    assert dominant.item() == -1 and summed.item() == -1
    # Unknown predictions stay in the class union, rather than being removed.
    cm = ebm.confusion(np.array([[0, 0]]), np.array([[0, -1]]), CLASSES)
    assert ebm.confusion_metrics(cm, [str(i) for i in range(CLASSES)])['foreground_miou'] == .5
    return dict(toy_only=True, split_slot_readout_difference=True, unknown_mass_preserved=True,
                coverage_gate_and_mismatch_rejection=True, majority_tie_and_unused_slot=True,
                TRAIN_only_NPZ_calibration_TEST_readout_and_errors=True, alpha_does_not_select_GT_support=True,
                global_radiance_permutation_EM=True, zero_prediction_unknown=True, unknown_union_penalty=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--clean-dataset', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--reference-cache', type=Path, default=CACHE)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        write_json(CACHE.parent / 'score_materials_selftest.json', dict(result, scorer_sha256=sha(__file__)))
        print(json.dumps(result))
        return
    if args.eval_dir is None or args.dataset is None:
        parser.error('--eval-dir and --dataset are required')
    dataset, output = args.dataset.resolve(), args.eval_dir.resolve()
    clean = args.clean_dataset or dataset.with_name(dataset.name.replace('_noisy', '_clean'))
    clean = clean.resolve()
    if clean == dataset:
        raise ValueError('Clean EM reference must be the clean sibling, not noisy observations')
    protocol = json.loads((dataset / 'material_class_protocol.json').read_text())
    names = protocol['primary_classes']
    if len(names) != CLASSES or protocol['material_id_to_primary_class'] != list(range(CLASSES)):
        raise ValueError('Scorer requires the frozen identity eight-class NVIDIA protocol')
    if protocol.get('surface_material_evaluation_minimum_coverage') != .5:
        raise ValueError('Material coverage threshold changed')
    splits = split_views(dataset)
    if split_views(clean) != splits:
        raise ValueError('Clean/noisy split mismatch')
    predictions = args.predictions
    if predictions is None:
        choices = [output / 'predictions', output / 'prediction_materials']
        found = [path for path in choices if path.is_dir()]
        if len(found) != 1:
            raise ValueError('Specify --predictions when neither or both prediction directories exist')
        predictions = found[0]
    predictions = predictions.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'benchmark_material_evaluation.json'
    source_before = dict(protocol=sha(dataset / 'material_class_protocol.json'),
                         splits={key: sha(dataset / f'transforms_{key}.json') for key in splits})
    mapping, calibration = calibrate(dataset, predictions, splits['train'], names)
    # Persist the global map before even opening a TEST prediction.
    write_json(report_path.with_suffix('.calibration.json'), calibration)
    print('Frozen TRAIN mapping:', mapping.tolist(), flush=True)
    evaluation = evaluate(dataset, predictions, mapping, splits['test'], names)
    references, counts, reference_receipt = training_references(dataset, clean, splits['train'], names, args.reference_cache)
    em = endmember_error(output / 'endmembers_radiance.npy', references, counts, names, reference_receipt)
    source_after = dict(protocol=sha(dataset / 'material_class_protocol.json'),
                        splits={key: sha(dataset / f'transforms_{key}.json') for key in splits})
    if source_after != source_before:
        raise ValueError('Frozen protocol changed during scoring')
    report = dict(schema_version=1, scene_id='nvidia_hq_v2', evaluator=str(Path(__file__).resolve()),
                  evaluator_sha256=sha(__file__), shared_rule_source_sha256=sha(Path(ebm.__file__)),
                  dataset=str(dataset), clean_dataset=str(clean), predictions=str(predictions), class_names=names,
                  calibration=calibration, evaluation=evaluation, endmember_EM=em,
                  endmembers_radiance_file=str(output / 'endmembers_radiance.npy'),
                  gt_dictionary_reference=dict(defined=False, score=None,
                      reason='A fixed first-surface material dictionary does not model glass-reflected source spectra, GGX highlights and atmospheric haze. No full eight-class NNLS upper-bound score is generated.'),
                  protocol_source_sha256=source_before, predictions_retained=True,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'endmember_EM.json', em)
    write_json(report_path, report)
    write_json(output / 'score_done.json', dict(status='complete', evaluator_output=str(report_path),
                                              report_sha256=sha(report_path), predictions_deleted=False,
                                              prediction_files=len(splits['train']) + len(splits['test']),
                                              calibration_sha256=calibration['sha256'], scene_id='nvidia_hq_v2'))
    metrics = evaluation['label_spaces'][SPACE]
    print(json.dumps(dict(mapping=mapping.tolist(),
        dominant_miou=metrics['mapped_dominant_slot_material_agreement']['foreground_miou'],
        class_sum_miou=metrics['summed_class_abundance_material_agreement']['foreground_miou'],
        composition=metrics['class_abundance_errors'], EM_degrees=em['mean_SAM_degrees'])), flush=True)


if __name__ == '__main__':
    main()

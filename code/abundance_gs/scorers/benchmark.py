#!/usr/bin/env python3
"""Train-calibrated many-to-one material evaluation for any decoder K (6- and 8-class label spaces).

Every learned endmember is mapped to the material class it dominates most often on the GT foreground of the TRAIN
views (majority vote); the map is written before any TEST prediction is read and is not refitted. TEST: mIoU and
pixel accuracy of the mapped dominant endmember and of the argmax of the class-summed abundance, plus abundance
errors. Endmember SAM in radiance space (EM) is a separate diagnostic.

Input: predictions/<stem>_abundance.npz of stage eval, `abundance` [K,H,W]. `alpha` is not a material vote; the
foreground is the GT coverage mask. Prediction and GT footprints must match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PRIMARY_MAPPING = np.array([0, 1, 2, 3, 1, 1, 4, 5], np.int16)
PRIMARY_NAMES = ['travertine', 'marble_granite_stone_group', 'grass',
                 'terracotta', 'sand', 'wood']
SECONDARY_NAMES = ['travertine', 'white_marble', 'grass', 'terracotta',
                   'pink_marble', 'granite', 'sand', 'wood']


def class_names_for_dataset(dataset):
    """Read frozen parent-class names without changing class IDs or mappings."""
    path = Path(dataset) / 'material_class_protocol.json'
    spec = json.loads(path.read_text()) if path.exists() else {}
    names = {'primary6': spec.get('primary_classes', PRIMARY_NAMES),
             'secondary8': spec.get('secondary_classes', SECONDARY_NAMES)}
    for space, count in [('primary6', 6), ('secondary8', 8)]:
        value = names[space]
        if (not isinstance(value, list) or len(value) != count
                or any(not isinstance(v, str) or not v for v in value)
                or len(set(value)) != count):
            raise ValueError(f'Protocol {space} must contain {count} unique class names')
    if 'material_id_to_primary_class' in spec and not np.array_equal(
            spec['material_id_to_primary_class'], PRIMARY_MAPPING):
        raise ValueError('Protocol material IDs disagree with the frozen6/8 class mapping')
    return names


def normalize_abundance(array, layout='chw'):
    a = np.asarray(array, np.float32)
    if a.ndim != 3:
        raise ValueError('Contribution map must be a three-dimensional K,H,W array')
    if layout == 'hwc':
        a = np.moveaxis(a, -1, 0)
    if not np.isfinite(a).all() or np.min(a) < -1e-7:
        raise ValueError('Contribution maps must be finite and nonnegative')
    a = np.maximum(a, 0)
    total = a.sum(0)
    if not np.isfinite(total).all():
        raise ValueError('Contribution sum overflow')
    return a / np.maximum(total[None], 1e-12), total > 1e-12


def add_votes(votes, abundance, labels):
    """Each foreground pixel casts one vote for its dominant learned channel."""
    if abundance.shape[1:] != labels.shape:
        raise ValueError('Prediction and GT footprints must match; no implicit resizing')
    active = abundance.sum(0) > 1e-12
    valid = (labels >= 0) & active
    dominant = abundance.argmax(0)
    channels, classes = votes.shape
    votes += np.bincount((dominant[valid] * classes + labels[valid]).reshape(-1),
                         minlength=channels * classes).reshape(channels, classes)


def mapping_from_votes(votes):
    mapping = votes.argmax(1).astype(np.int16)
    mapping[votes.sum(1) == 0] = -1
    return mapping


def aggregate_channels(abundance, mapping, classes):
    """Sum same-class channels; preserve unassigned mass without renormalizing."""
    result = np.zeros((classes, *abundance.shape[1:]), np.float32)
    unassigned = np.zeros(abundance.shape[1:], np.float32)
    for channel, label in enumerate(mapping):
        if label < 0:
            unassigned += abundance[channel]
        else:
            result[label] += abundance[channel]
    return result, unassigned


def mapped_labels(abundance, mapping, class_abundance, unassigned):
    active = abundance.sum(0) > 1e-12
    dominant = np.where(active, mapping[abundance.argmax(0)], -1)
    merged = np.concatenate([class_abundance, unassigned[None]], axis=0).argmax(0)
    merged = np.where(active & (merged < class_abundance.shape[0]), merged, -1)
    return dominant, merged


def confusion(labels, predicted, classes):
    valid = labels >= 0
    p = np.where(predicted[valid] >= 0, predicted[valid], classes)
    return np.bincount((labels[valid] * (classes + 1) + p).reshape(-1),
                       minlength=classes * (classes + 1)).reshape(classes, classes + 1)


def confusion_metrics(matrix, names):
    classes = len(names)
    true = matrix.sum(1)
    pred = matrix[:, :classes].sum(0)
    intersection = np.diag(matrix[:, :classes])
    union = true + pred - intersection
    iou = np.divide(intersection, union, out=np.zeros(classes, float), where=union > 0)
    present = union > 0
    return {
        'foreground_pixel_accuracy': float(intersection.sum() / max(matrix.sum(), 1)),
        'foreground_miou': float(iou[present].mean()) if present.any() else None,
        'iou_per_class': [float(iou[k]) if present[k] else None for k in range(classes)],
        'class_names': names,
        'gt_support_pixels': true.tolist(),
        'predicted_class_pixels': pred.tolist(),
        'unassigned_prediction_pixels': int(matrix[:, -1].sum()),
        'confusion_matrix_gt_rows_pred_columns': matrix.tolist(),
        'confusion_column_names': names + ['unassigned'],
        'aggregation': 'Pooled foreground confusion over all scored views; absent unions omitted from mIoU.',
    }


def read_prediction(directory, stem, suffix, key, layout):
    path = Path(directory) / f'{stem}{suffix}'
    if path.suffix == '.npz':
        with np.load(path) as z:
            if key not in z:
                raise KeyError(f'{path} lacks {key}; available keys: {z.files}')
            raw = z[key]
    else:
        raw = np.load(path)
    return normalize_abundance(raw, layout)[0]


def stems_for_split(dataset, split):
    frames = json.loads((Path(dataset) / f'transforms_{split}.json').read_text())['frames']
    return [Path(frame['file_path']).stem for frame in frames]


def load_labels(dataset, stem):
    with np.load(Path(dataset) / 'gt' / f'{stem}.npz') as z:
        labels = {'primary6': z['material_merged'], 'secondary8': z['material']}
        foreground = z['coverage'] > 0
    if any(not np.array_equal(v >= 0, foreground) for v in labels.values()):
        raise ValueError('Material labels and explicit GT coverage foreground differ')
    return labels


def calibrate(dataset, prediction_directory, suffix='_abundance.npz',
              key='abundance', layout='chw'):
    """Fit both label spaces globally from all training views only."""
    train = stems_for_split(dataset, 'train')
    held_out = set(stems_for_split(dataset, 'val') + stems_for_split(dataset, 'test'))
    if set(train) & held_out:
        raise ValueError('Training calibration overlaps validation/test')
    votes = None
    mass = None
    for stem in train:
        a = read_prediction(prediction_directory, stem, suffix, key, layout)
        labels = load_labels(dataset, stem)
        if votes is None:
            votes = {'primary6': np.zeros((a.shape[0], 6), np.int64),
                     'secondary8': np.zeros((a.shape[0], 8), np.int64)}
            mass = np.zeros(a.shape[0], np.float64)
        if a.shape[0] != len(mass):
            raise ValueError('Learned channel count changes across views')
        foreground = labels['secondary8'] >= 0
        mass += a[:, foreground].sum(1, dtype=np.float64)
        for space in votes:
            add_votes(votes[space], a, labels[space])
    if votes is None:
        raise ValueError('No training calibration views')
    mappings = {space: mapping_from_votes(table) for space, table in votes.items()}
    payload = {
        'algorithm': 'global many-to-one dominant-channel majority vote',
        'calibration_split': 'train', 'calibration_views': train,
        'held_out_gt_used_for_mapping': False,
        'foreground_mask': 'GT coverage > 0, verified equivalent to material ID >= 0',
        'K': len(mass), 'mapping_tie_rule': 'lowest class ID among tied positive votes',
        'unassigned_channel_rule': 'No dominant foreground training votes -> -1; no invented class.',
        'channel_foreground_training_abundance_mass': mass.tolist(),
        'empty_training_channels': np.flatnonzero(mass <= 1e-12).tolist(),
        'label_spaces': {space: {
            'channel_to_class': mappings[space].tolist(),
            'votes': votes[space].tolist(),
            'unassigned_channels': np.flatnonzero(mappings[space] < 0).tolist(),
            'nonempty_but_never_dominant_channels': np.flatnonzero((mass > 1e-12) & (mappings[space] < 0)).tolist(),
        } for space in mappings},
    }
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return mappings, payload


def sam_matrix(learned, references):
    denominator = np.linalg.norm(learned, axis=1)[:, None] * np.linalg.norm(references, axis=1)[None]
    cosine = (learned @ references.T) / np.maximum(denominator, 1e-30)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def endpoint_diagnostics(dataset, learned, mappings, calibration=None):
    """Class-conditioned nearest radiance reference; references may be reused."""
    dataset = Path(dataset)
    if (dataset / 'gt/endmembers_radiance_samples.npy').exists():
        from abundance_gs.scorers.intraclass_diagnostics import sample_endpoint_diagnostics
        mass = calibration.get('channel_foreground_training_abundance_mass') if calibration else None
        return sample_endpoint_diagnostics(dataset, learned, mappings, mass)
    direct = np.load(dataset / 'gt/endmembers_radiance.npy').T.astype(float)
    sky_path = dataset / 'gt/endmembers_radiance_sky.npy'
    sky = np.load(sky_path).T.astype(float) if sky_path.exists() else direct.copy()
    learned = np.asarray(learned, float)
    if learned.ndim != 2 or learned.shape[1] != direct.shape[1]:
        raise ValueError('Learned spectra must be K x bands in radiance space')
    if not np.isfinite(learned).all():
        raise ValueError('Nonfinite learned endmembers')
    references = np.concatenate([direct, sky])
    angles = sam_matrix(learned, references)
    nonzero = np.linalg.norm(learned, axis=1) > 1e-12
    result = {}
    for space, mapping in mappings.items():
        if len(mapping) != len(learned):
            raise ValueError('Endmember K differs from calibrated contribution K')
        ref_classes = PRIMARY_MAPPING if space == 'primary6' else np.arange(8)
        rows = []
        for channel, label in enumerate(mapping):
            row = {'channel': channel, 'mapped_class': int(label), 'nonzero_spectrum': bool(nonzero[channel])}
            if label >= 0 and nonzero[channel]:
                materials = np.flatnonzero(ref_classes == label)
                nominal = int(materials[np.argmin(angles[channel, materials])])
                candidates = np.concatenate([materials, materials + 8])
                best = int(candidates[np.argmin(angles[channel, candidates])])
                row.update(nominal_direct_SAM_degrees=float(angles[channel, nominal]),
                           nominal_direct_reference_material_id=nominal,
                           nearest_direct_or_sky_SAM_degrees=float(angles[channel, best]),
                           nearest_reference_material_id=best % 8,
                           nearest_reference_context='direct' if best < 8 else 'sky')
            else:
                row.update(nominal_direct_SAM_degrees=None, nearest_direct_or_sky_SAM_degrees=None)
            rows.append(row)
        coverage = []
        for ref in range(16):
            candidates = np.flatnonzero((mapping == ref_classes[ref % 8]) & nonzero)
            best = int(candidates[np.argmin(angles[candidates, ref])]) if len(candidates) else None
            coverage.append({'reference_material_id': ref % 8,
                             'context': 'direct' if ref < 8 else 'sky',
                             'closest_mapped_learned_channel': best,
                             'SAM_degrees': float(angles[best, ref]) if best is not None else None})
        result[space] = {'per_channel': rows, 'reference_coverage': coverage,
                         'one_to_one_constraint': False}
    return {
        'interpretation': 'Diagnostic agreement with nominal direct/sky radiance references. Distinct illumination has no unique globally invariant eight-radiance basis. No Hungarian assignment; arbitrary K and reusable references.',
        'label_spaces': result,
    }


def evaluate(dataset, prediction_directory, mappings, split='test',
             suffix='_abundance.npz', key='abundance', layout='chw'):
    states = {}
    names = class_names_for_dataset(dataset)
    sample_variant = (Path(dataset)/'gt/endmembers_radiance_samples.npy').exists()
    targets = ['area', 'nominal_irradiance_weighted']
    if sample_variant:
        targets.append('broadband_radiance_weighted')
    for space, mapping in mappings.items():
        classes = len(names[space])
        states[space] = {
            'dominant': np.zeros((classes, classes + 1), np.int64),
            'summed': np.zeros((classes, classes + 1), np.int64),
            'squared': {target: np.zeros(classes, float) for target in targets},
            'absolute': {target: np.zeros(classes, float) for target in targets},
            'pixels': 0, 'unassigned_mass_sum': 0.,
        }
    views = stems_for_split(dataset, split)
    for stem in views:
        a = read_prediction(prediction_directory, stem, suffix, key, layout)
        with np.load(Path(dataset) / 'gt' / f'{stem}.npz') as z:
            labels = {'primary6': z['material_merged'], 'secondary8': z['material']}
            foreground = z['coverage'] > 0
            area = {'primary6': np.moveaxis(z['abundance_area_merged'], -1, 0),
                    'secondary8': np.moveaxis(z['abundance_area'], -1, 0)}
            rad8 = np.moveaxis(z['abundance_irradiance_weighted'], -1, 0)
            if sample_variant:
                broadband = {'primary6': np.moveaxis(z['abundance_radiance_weighted_class_merged'], -1, 0),
                             'secondary8': np.moveaxis(z['abundance_radiance_weighted_class'], -1, 0)}
        if any(not np.array_equal(v >= 0, foreground) for v in labels.values()):
            raise ValueError('Scoring foreground mask disagrees with explicit GT coverage')
        radiometric = {'secondary8': rad8,
                       'primary6': np.stack([rad8[PRIMARY_MAPPING == c].sum(0) for c in range(6)])}
        truths = {'area': area, 'nominal_irradiance_weighted': radiometric}
        if sample_variant:
            truths['broadband_radiance_weighted'] = broadband
        for space, mapping in mappings.items():
            if a.shape[0] != len(mapping) or a.shape[1:] != labels[space].shape:
                raise ValueError('Scored predictions differ from calibrated K or GT footprint')
            state = states[space]
            class_ab, unknown = aggregate_channels(a, mapping, len(names[space]))
            dominant, summed = mapped_labels(a, mapping, class_ab, unknown)
            state['dominant'] += confusion(labels[space], dominant, len(names[space]))
            state['summed'] += confusion(labels[space], summed, len(names[space]))
            valid = labels[space] >= 0
            state['pixels'] += int(valid.sum())
            state['unassigned_mass_sum'] += float(unknown[valid].sum(dtype=np.float64))
            for target in targets:
                truth = truths[target][space]
                if truth.shape != class_ab.shape or not np.isfinite(truth).all() or np.min(truth) < -1e-7:
                    raise ValueError(f'Invalid {target} class-abundance GT footprint or values')
                delta = (class_ab[:, valid] - truth[:, valid]).astype(np.float64)
                state['squared'][target] += np.sum(delta ** 2, axis=1)
                state['absolute'][target] += np.sum(abs(delta), axis=1)
    result = {}
    for space, state in states.items():
        n = max(state['pixels'], 1)
        classes = len(names[space])
        result[space] = {
            'mapped_dominant_slot_material_agreement': confusion_metrics(state['dominant'], names[space]),
            'summed_class_abundance_material_agreement': confusion_metrics(state['summed'], names[space]),
            'mean_unassigned_foreground_abundance_mass': state['unassigned_mass_sum'] / n,
            'class_abundance_errors': {target: {
                'RMSE': float(np.sqrt(state['squared'][target].sum() / (n * classes))),
                'MAE': float(state['absolute'][target].sum() / (n * classes)),
                'RMSE_per_class': np.sqrt(state['squared'][target] / n).tolist(),
                'MAE_per_class': (state['absolute'][target] / n).tolist(),
            } for target in state['squared']},
        }
    payload = {'split': split, 'views': views, 'mapping_refitted_on_scored_views': False,
            'foreground_mask': 'GT coverage > 0 (equivalent to material ID >= 0)',
            'label_spaces': result,
            'abundance_note': 'All mapped channels are summed; unassigned mass is retained separately and is never redistributed. Area and nominal irradiance-weighted targets are distinct. The latter is a coefficient convention, not an exact eight-endmember reconstruction of the distinct-sky variant.',
            'abundance_statistic_scope': 'Errors are diagnostic target agreements, not proof of uniquely recovered physical fractions. Unit-normalized endmember weights incorporate spectral norms. Abundance exports of this package composite per-hit softmax abundance without per-hit intensity, so their statistic differs from radiance-weighted coefficients; the chosen prediction key/layout must be disclosed.'}
    if sample_variant:
        payload['abundance_note'] = 'All mapped channels are summed; unassigned mass is retained separately. Three targets are distinct: foreground-conditional area; nominal scalar illumination coefficients; actual broadband class radiance fractions weighted by each measured sample/direct/sky400--1000nm radiance integral. None defines a unique eight-radiance endmember reconstruction.'
        payload['broadband_target_definition'] = 'Sum coefficients_context_sample times integral400--1000nm rho_sample(lambda)*E_context(lambda)/pi for each class, then normalize by total foreground radiance. No learned/model/test-dependent normalization.'
    return payload


def self_check():
    """K16 split illumination plus an unseen test-only channel regression."""
    label8 = np.tile(np.arange(8, dtype=np.int16), (2, 1))
    label6 = PRIMARY_MAPPING[label8]
    votes8 = np.zeros((16, 8), np.int64)
    votes6 = np.zeros((16, 6), np.int64)
    for context in [0, 1]:
        a = np.zeros((16, 2, 8), np.float32)
        for material in range(8):
            a[2 * material + context, :, material] = 1
        add_votes(votes8, a, label8)
        add_votes(votes6, a, label6)
    mappings = {'secondary8': mapping_from_votes(votes8), 'primary6': mapping_from_votes(votes6)}
    assert np.array_equal(mappings['secondary8'], np.repeat(np.arange(8), 2))
    assert np.array_equal(mappings['primary6'], np.repeat(PRIMARY_MAPPING, 2))
    test = np.zeros((16, 2, 8), np.float32)
    for material in range(8):
        test[2 * material:2 * material + 2, :, material] = .5
    for space, labels in [('secondary8', label8), ('primary6', label6)]:
        classes = 8 if space == 'secondary8' else 6
        summed, unknown = aggregate_channels(test, mappings[space], classes)
        dominant, class_dominant = mapped_labels(test, mappings[space], summed, unknown)
        truth = np.eye(classes, dtype=np.float32)[labels].transpose(2, 0, 1)
        assert np.array_equal(dominant, labels) and np.array_equal(class_dominant, labels)
        assert np.array_equal(summed, truth) and np.all(unknown == 0)
    # An empty training channel must remain unknown if it activates at test.
    mapping17 = mapping_from_votes(np.vstack([votes8, np.zeros((1, 8), np.int64)]))
    assert mapping17[-1] == -1
    unknown_test = np.vstack([test, np.zeros((1, 2, 8), np.float32)])
    unknown_test[:, 0, 0] = 0
    unknown_test[-1, 0, 0] = 1
    summed, unknown = aggregate_channels(unknown_test, mapping17, 8)
    dominant, class_dominant = mapped_labels(unknown_test, mapping17, summed, unknown)
    assert dominant[0, 0] == -1 and class_dominant[0, 0] == -1 and unknown[0, 0] == 1
    matrix = confusion(label8, dominant, 8)
    assert matrix[0, -1] == 1
    zero = np.zeros_like(test)
    zero_votes = np.zeros_like(votes8)
    add_votes(zero_votes, zero, label8)
    assert not zero_votes.any()
    zero_sum, zero_unknown = aggregate_channels(zero, mappings['secondary8'], 8)
    zero_dom, zero_class = mapped_labels(zero, mappings['secondary8'], zero_sum, zero_unknown)
    assert np.all(zero_dom == -1) and np.all(zero_class == -1)
    partial = unknown_test.copy()
    partial[:, 0, 0] = 0
    partial[0, 0, 0] = .6
    partial[16, 0, 0] = .4
    partial_sum, partial_unknown = aggregate_channels(partial, mapping17, 8)
    assert np.isclose(partial_sum[0, 0, 0], .6) and np.isclose(partial_unknown[0, 0], .4)
    # No spectral permutation: both exact sun/sky slots can use their material's
    # references independently, even though the number of slots exceeds eight.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='material-evaluator-selfcheck-') as temporary:
        path = Path(temporary)
        (path / 'gt').mkdir()
        direct = np.eye(64, dtype=np.float32)[:8]
        sky = np.eye(64, dtype=np.float32)[8:16]
        np.save(path / 'gt/endmembers_radiance.npy', direct.T)
        np.save(path / 'gt/endmembers_radiance_sky.npy', sky.T)
        learned = np.stack([bank[material] for material in range(8) for bank in [direct, sky]])
        diagnostic = endpoint_diagnostics(path, learned, mappings)
        for space in mappings:
            rows = diagnostic['label_spaces'][space]['per_channel']
            assert all(row['nearest_direct_or_sky_SAM_degrees'] == 0 for row in rows)
        # Exercise the NPZ reader, split calibration, foreground mask,
        # pooled scorer, and abundance targets with tiny artificial fixtures.
        predictions = path / 'predictions'
        predictions.mkdir()
        truth8 = np.eye(8, dtype=np.float32)[label8]
        truth6 = np.eye(6, dtype=np.float32)[label6]
        for index in range(3):
            stem = f'f_{index:04d}'
            np.savez(path / 'gt' / f'{stem}.npz', material=label8,
                     material_merged=label6, coverage=np.ones(label8.shape, np.float32),
                     abundance_area=truth8, abundance_area_merged=truth6,
                     abundance_irradiance_weighted=truth8,
                     abundance_radiance_weighted_class=truth8,
                     abundance_radiance_weighted_class_merged=truth6)
            prediction = test.copy()
            if index < 2:
                prediction.fill(0)
                for material in range(8):
                    prediction[2 * material + index, :, material] = 1
            np.savez(predictions / f'{stem}_abundance.npz', abundance=prediction * .3,
                     alpha=np.full(label8.shape, .3, np.float32))
        for split, indices in [('train', [0, 1]), ('val', []), ('test', [2])]:
            (path / f'transforms_{split}.json').write_text(json.dumps({
                'frames': [{'file_path': f'./images/f_{index:04d}'} for index in indices]}))
        fitted, calibration = calibrate(path, predictions)
        scored = evaluate(path, predictions, fitted)
        assert calibration['calibration_views'] == ['f_0000', 'f_0001']
        assert not calibration['held_out_gt_used_for_mapping']
        for space in fitted:
            metrics = scored['label_spaces'][space]
            assert metrics['mapped_dominant_slot_material_agreement']['foreground_miou'] == 1
            assert metrics['summed_class_abundance_material_agreement']['foreground_miou'] == 1
            assert all(error['RMSE'] == 0 for error in metrics['class_abundance_errors'].values())
        # Parent semantics may broaden with multiple measured specimens while
        # all frozen numeric class IDs and calibration rules remain unchanged.
        renamed = {'primary_classes': [name+'_parent' for name in PRIMARY_NAMES],
                   'secondary_classes': [name+'_parent' for name in SECONDARY_NAMES],
                   'material_id_to_primary_class': PRIMARY_MAPPING.tolist()}
        (path/'material_class_protocol.json').write_text(json.dumps(renamed))
        renamed_score = evaluate(path, predictions, fitted)
        assert renamed_score['label_spaces']['secondary8']['mapped_dominant_slot_material_agreement']['class_names'] == renamed['secondary_classes']
        # Exercise dispatch to the new measured sample/context branch with
        # train-only cached support. The original nominal branch was tested first.
        np.save(path/'gt/endmembers_radiance_samples.npy', direct.T)
        np.save(path/'gt/endmembers_radiance_sky_samples.npy', sky.T)
        np.save(path/'gt/spectral_sample_material_ids.npy', np.arange(8))
        np.savez(path/'gt/spectral_sample_training_weights.npz',
                 train_frame_stems=np.array(['f_0000', 'f_0001']),
                 sample_area_sum=np.ones(8), direct_coefficient_sum=np.ones(8),
                 sky_coefficient_sum=np.ones(8), direct_radiance_power=np.ones(8),
                 sky_radiance_power=np.ones(8))
        sample_diagnostic = endpoint_diagnostics(path, learned, fitted, calibration)
        sample_score = evaluate(path, predictions, fitted)
        for space in fitted:
            assert len(sample_score['label_spaces'][space]['class_abundance_errors']) == 3
            assert sample_score['label_spaces'][space]['class_abundance_errors']['broadband_radiance_weighted']['RMSE'] == 0
            sample_metrics = sample_diagnostic['label_spaces'][space]
            assert sample_metrics['reference_recall_training_radiance_weighted']['within_angle_weight_fraction']['1'] == 1
            assert sample_metrics['learned_channel_precision_training_mass_weighted']['represented_weight_fraction'] == 1
    return {'toy_only': True, 'K16_two_channels_per_material': 'passed',
            'summed_material_abundance_exact': True, 'primary6_and_secondary8': 'passed',
            'K17_empty_then_test_only_channel_remains_unknown': 'passed',
            'foreground_zero_sum_prediction_remains_unknown': 'passed',
            'partial_unassigned_mass_not_renormalized': 'passed',
            'sun_shadow_endmember_reference_reuse_without_one_to_one_constraint': 'passed',
            'NPZ_train_calibration_and_held_out_scoring_fixture': 'passed',
            'protocol_parent_class_names_without_ID_changes': 'passed',
            'sample_bank_dispatch_train_weighted_precision_and_reference_recall': 'passed',
            'sample_variant_distinct_area_nominal_and_broadband_abundance_targets': 'passed',
            'no_learned_model_or_dataset_score_reported': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--train-predictions', type=Path)
    parser.add_argument('--prediction-suffix', default='_abundance.npz')
    parser.add_argument('--prediction-key', default='abundance')
    parser.add_argument('--prediction-layout', choices=['chw', 'hwc'], default='chw')
    parser.add_argument('--endmembers', type=Path, help='NPY K x bands radiance endmembers; transpose explicitly before export if needed.')
    parser.add_argument('--split', choices=['test', 'val'], default='test')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if not args.dataset or not args.predictions:
        parser.error('--dataset and --predictions are required for model evaluation')
    train_predictions = args.train_predictions or args.predictions
    mappings, calibration = calibrate(args.dataset, train_predictions,
                                      args.prediction_suffix, args.prediction_key, args.prediction_layout)
    output = args.output or args.predictions / 'benchmark_material_evaluation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping_path = output.with_suffix('.mapping.json')
    # Persist the complete training-only mapping before loading/scoring any
    # held-out prediction or GT, making the calibration stage reviewable.
    mapping_path.write_text(json.dumps(calibration, indent=2) + '\n')
    result = evaluate(args.dataset, args.predictions, mappings, args.split,
                      args.prediction_suffix, args.prediction_key, args.prediction_layout)
    payload = {'calibration': calibration, 'calibration_freeze_file': str(mapping_path.resolve()),
               'evaluation': result,
               'protocol_lock': json.loads((args.dataset / 'protocol_lock.json').read_text())}
    if args.endmembers:
        payload['endmember_diagnostics'] = endpoint_diagnostics(args.dataset, np.load(args.endmembers), mappings, calibration)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()

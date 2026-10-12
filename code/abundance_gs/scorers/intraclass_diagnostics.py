"""Endmember diagnostics against measured-sample radiance banks, used by benchmark.endpoint_diagnostics for
datasets that ship gt/endmembers_radiance_samples.npy. Diagnostics only; not part of the material score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRIMARY_MAPPING = np.array([0, 1, 2, 3, 1, 1, 4, 5], np.int16)


def split_stems(dataset, split):
    frames = json.loads((Path(dataset) / f'transforms_{split}.json').read_text())['frames']
    return [Path(frame['file_path']).stem for frame in frames]


def load_sample_banks(dataset):
    gt = Path(dataset) / 'gt'
    direct = np.load(gt / 'endmembers_radiance_samples.npy').T.astype(np.float64)
    sky = np.load(gt / 'endmembers_radiance_sky_samples.npy').T.astype(np.float64)
    material_ids = np.load(gt / 'spectral_sample_material_ids.npy').astype(np.int64)
    if direct.ndim != 2 or sky.shape != direct.shape or material_ids.shape != (len(direct),):
        raise ValueError('Sample banks must be bands x samples and material IDs samples')
    if (not np.isfinite(direct).all() or not np.isfinite(sky).all()
            or np.min(direct) < 0 or np.min(sky) < 0
            or np.any(np.linalg.norm(direct, axis=1) <= 0)
            or np.any(np.linalg.norm(sky, axis=1) <= 0)
            or np.any(material_ids < 0) or np.any(material_ids >= 8)):
        raise ValueError('Invalid sample radiance bank or parent-material IDs')
    return direct, sky, material_ids


def load_training_weights(dataset, direct, sky):
    """Use a train-only cache, otherwise accumulate actual saved TRAIN GT."""
    dataset = Path(dataset)
    stems = split_stems(dataset, 'train')
    held_out = set(split_stems(dataset, 'val') + split_stems(dataset, 'test'))
    if not stems or len(set(stems)) != len(stems) or set(stems) & held_out:
        raise ValueError('Invalid or overlapping training split')
    samples = len(direct)
    cache = dataset / 'gt/spectral_sample_training_weights.npz'
    powers = None
    if cache.exists():
        with np.load(cache) as z:
            if 'train_frame_stems' not in z or z['train_frame_stems'].tolist() != stems:
                raise ValueError('Sample-weight cache must explicitly match training frame order')
            values = [z[name].astype(np.float64) for name in
                      ['sample_area_sum', 'direct_coefficient_sum', 'sky_coefficient_sum']]
            if 'direct_radiance_power' in z and 'sky_radiance_power' in z:
                powers = [z['direct_radiance_power'].astype(np.float64),
                          z['sky_radiance_power'].astype(np.float64)]
        source = 'Cached sums, all pixels of explicitly listed training views only'
    else:
        values = [np.zeros(samples, np.float64) for _ in range(3)]
        for stem in stems:
            with np.load(dataset / 'gt' / f'{stem}.npz') as z:
                for accumulator, key in zip(values, ['sample_area_coverage',
                        'coefficients_direct_samples', 'coefficients_sky_samples']):
                    array = z[key]
                    if array.ndim != 3 or array.shape[-1] != samples:
                        raise ValueError(f'{stem}: unexpected {key} shape')
                    if not np.isfinite(array).all() or np.min(array) < 0:
                        raise ValueError(f'{stem}: invalid {key}')
                    accumulator += array.sum(axis=(0, 1), dtype=np.float64)
        source = 'Sums computed from all pixels of saved training GT only'
    if any(v.shape != (samples,) or not np.isfinite(v).all() or np.min(v) < 0 for v in values):
        raise ValueError('Invalid sample training support weights')
    if powers is None:
        powers = [np.linalg.norm(direct, axis=1), np.linalg.norm(sky, axis=1)]
        convention = 'Coefficient sum times 64-band radiance-spectrum L2 norm; not a broadband energy integral'
    else:
        if any(v.shape != (samples,) or not np.isfinite(v).all() or np.min(v) <= 0 for v in powers):
            raise ValueError('Invalid broadband radiance powers')
        convention = 'Coefficient sum times measured-spectrum 400--1000nm radiance integral (W m^-2 sr^-1)'
    area, coeff_direct, coeff_sky = values
    return {
        'sample_area_sum': area,
        'direct_coefficient_sum': coeff_direct,
        'sky_coefficient_sum': coeff_sky,
        'context_radiance_weight': np.concatenate([coeff_direct*powers[0], coeff_sky*powers[1]]),
        'context_equal_area_weight': np.tile(area*.5, 2),
        'metadata': {'calibration_split': 'train', 'training_views': stems,
                     'held_out_gt_used_for_weights': False, 'support_source': source,
                     'radiance_weight_convention': convention,
                     'area_context_weight_convention': 'Half of each sample area assigned to each reference context; an area diagnostic, not observed context energy.'},
    }


def spectral_angles(learned, references):
    denominator = np.linalg.norm(learned, axis=1)[:, None]*np.linalg.norm(references, axis=1)[None]
    cosine = learned @ references.T / np.maximum(denominator, 1e-30)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def weighted_angle_summary(angles, weights):
    """Report missing support explicitly; a missing nonnegative vector costs90deg."""
    weights = np.asarray(weights, float)
    values = np.array([np.nan if value is None else value for value in angles], float)
    total = float(weights.sum())
    if total <= 0:
        return {'total_weight': 0., 'weighted_mean_SAM_degrees': None,
                'weighted_mean_SAM_with_missing_90deg': None,
                'represented_weight_fraction': None,
                'within_angle_weight_fraction': {str(t): None for t in [1, 3, 5, 10]}}
    represented = np.isfinite(values)
    support = float(weights[represented].sum())
    return {
        'total_weight': total,
        'weighted_mean_SAM_degrees': float(np.dot(values[represented], weights[represented])/support) if support > 0 else None,
        'weighted_mean_SAM_with_missing_90deg': float(np.dot(np.where(represented, values, 90.), weights)/total),
        'represented_weight_fraction': support/total,
        'within_angle_weight_fraction': {str(t): float(weights[represented & (values <= t)].sum()/total) for t in [1, 3, 5, 10]},
        'missing_angle_convention': 'Missing mapped nonzero spectrum retains its weight and costs90deg, the maximum angle of nonnegative spectra; it never contributes to within-angle recall.',
    }


def sample_endpoint_diagnostics(dataset, learned, mappings, channel_training_mass=None):
    """Reusable nearest references; arbitrary K, train-only many-to-one classes."""
    direct, sky, material_ids = load_sample_banks(dataset)
    learned = np.asarray(learned, np.float64)
    if (learned.ndim != 2 or learned.shape[1] != direct.shape[1]
            or not np.isfinite(learned).all() or np.min(learned) < 0):
        raise ValueError('Learned endmembers must be finite nonnegative K x bands radiance spectra')
    weights = load_training_weights(dataset, direct, sky)
    references = np.concatenate([direct, sky])
    samples = len(direct)
    ref_materials = np.tile(material_ids, 2)
    angles = spectral_angles(learned, references)
    nonzero = np.linalg.norm(learned, axis=1) > 1e-12
    if channel_training_mass is not None:
        channel_training_mass = np.asarray(channel_training_mass, float)
        if (channel_training_mass.shape != (len(learned),)
                or not np.isfinite(channel_training_mass).all() or np.min(channel_training_mass) < 0):
            raise ValueError('Channel weights must be training foreground normalized-abundance sums')
    result = {}
    for space, raw_mapping in mappings.items():
        mapping = np.asarray(raw_mapping, np.int64)
        if mapping.shape != (len(learned),):
            raise ValueError('Endmember K differs from calibrated contribution K')
        ref_classes = PRIMARY_MAPPING[ref_materials] if space == 'primary6' else ref_materials
        if space not in ['primary6', 'secondary8']:
            raise ValueError(f'Unknown label space {space}')
        classes = 6 if space == 'primary6' else 8
        if np.any(mapping < -1) or np.any(mapping >= classes):
            raise ValueError('Invalid mapped class')
        rows = []
        for channel, label in enumerate(mapping):
            row = {'channel': channel, 'mapped_class': int(label),
                   'nonzero_spectrum': bool(nonzero[channel])}
            candidates = np.flatnonzero(ref_classes == label) if label >= 0 and nonzero[channel] else np.array([], int)
            if len(candidates):
                best = int(candidates[np.argmin(angles[channel, candidates])])
                row.update(nearest_direct_or_sky_SAM_degrees=float(angles[channel, best]),
                           nearest_reference_sample_id=best % samples,
                           nearest_reference_material_id=int(ref_materials[best]),
                           nearest_reference_context='direct' if best < samples else 'sky')
            else:
                row['nearest_direct_or_sky_SAM_degrees'] = None
            if channel_training_mass is not None:
                row['foreground_training_abundance_mass'] = float(channel_training_mass[channel])
            rows.append(row)
        coverage = []
        for ref, label in enumerate(ref_classes):
            candidates = np.flatnonzero((mapping == label) & nonzero)
            best = int(candidates[np.argmin(angles[candidates, ref])]) if len(candidates) else None
            coverage.append({
                'reference_sample_id': ref % samples, 'reference_material_id': int(ref_materials[ref]),
                'context': 'direct' if ref < samples else 'sky',
                'closest_mapped_learned_channel': best,
                'SAM_degrees': float(angles[best, ref]) if best is not None else None,
                'training_sample_area_sum': float(weights['sample_area_sum'][ref % samples]),
                'training_context_coefficient_sum': float(weights['direct_coefficient_sum' if ref < samples else 'sky_coefficient_sum'][ref % samples]),
                'training_context_radiance_weight': float(weights['context_radiance_weight'][ref]),
            })
        reference_angles = [r['SAM_degrees'] for r in coverage]
        result[space] = {
            'per_channel': rows, 'reference_coverage': coverage, 'one_to_one_constraint': False,
            'reference_recall_training_radiance_weighted': weighted_angle_summary(reference_angles, weights['context_radiance_weight']),
            'reference_recall_training_area_context_equal_weighted': weighted_angle_summary(reference_angles, weights['context_equal_area_weight']),
            'learned_channel_precision_training_mass_weighted': weighted_angle_summary(
                [r['nearest_direct_or_sky_SAM_degrees'] for r in rows], channel_training_mass) if channel_training_mass is not None else None,
            'precision_weight_note': 'Training foreground unit-sum channel contribution mass. No test abundance used. Unassigned or zero-spectrum channels retain their weight.',
        }
    return {
        'interpretation': 'Diagnostic agreement with the frozen measured sample/direct/sky dictionary within each train-calibrated class. References are reusable; no unique eight-endmember truth or one-to-one matching is assumed. A pure direct/sky endpoint diagnostic does not by itself score every valid illumination mixture.',
        'sample_count': samples, 'context_reference_count': 2*samples,
        'weight_provenance': weights['metadata'],
        'reference_dictionary_frozen_before_model_training': True,
        'label_spaces': result,
    }

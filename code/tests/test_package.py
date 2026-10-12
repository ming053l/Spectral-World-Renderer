"""Unit tests of the abundance_gs package (CPU), plus an opt-in GPU smoke test.

Run in the package environment:  env/run.sh -1 tests/test_package.py
GPU smoke test (a 20-step train + eval + render + fps through abgs.py on a real pack):
  ABGS_SMOKE_GPU=<gpu> ABGS_SMOKE_PACK=<pack> ABGS_SMOKE_SOURCE=<source ckpt> ABGS_SMOKE_OUT=<empty dir> \
  [ABGS_SMOKE_CLUSTERS=<cache> ABGS_SMOKE_K=<K>] env/run.sh <gpu> tests/test_package.py GPUSmoke
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abundance_gs import cli, clusters, data, evaluate, inference, metrics, model, pipeline, records, scenes, train  # noqa: E402
from abundance_gs.defaults import PRESETS, Decoder                                            # noqa: E402


def fixture(root, bands=21, views=10, spatial=16):
    """Eight TRAIN plus two TEST views with three spectra in three quadrants."""
    root = Path(root)
    root.mkdir(parents=True)
    axis = np.linspace(0, 1, bands, dtype=np.float32)
    spectra = np.stack([.08 + .24 * axis, .10 + .24 * (1 - axis), .10 + .16 * np.sin(axis * np.pi) ** 2])
    entries = []
    for index in range(views):
        path = root / ('observed_%02d.npy' % index)
        cube = np.zeros((bands, spatial, spatial), np.float32)
        cube[:, :8, :8] = spectra[0, :, None, None]
        cube[:, :8, 8:] = spectra[1, :, None, None]
        cube[:, 8:, :8] = spectra[2, :, None, None]
        np.save(path, cube * (1 + .025 * index))
        entries.append(dict(name='v%02d' % index, split='train' if index < 8 else 'test', hsi_path=str(path),
                            width=spatial, height=spatial, fx=16., fy=16., cx=8., cy=8., c2w=np.eye(4).tolist()))
    (root / 'pack.json').write_text(json.dumps(dict(scene_id='package_test', bands=bands, views=entries)))
    return root


class CLITests(unittest.TestCase):
    def test_help_of_every_stage(self):
        for stage in [None, *cli.STAGES]:
            argv = [sys.executable, str(ROOT / 'abgs.py')] + ([stage] if stage else []) + ['--help']
            with self.subTest(stage=stage):
                result = subprocess.run(argv, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage', result.stdout)

    def test_every_extension_has_its_sources(self):
        from abundance_gs import ext
        for name in ext.SPECS:
            with self.subTest(extension=name):
                spec = ext.SPECS[name]()
                self.assertTrue(all(Path(s).is_file() for s in spec['sources']))
                self.assertEqual(len(ext.digest(name)), 64)


class ConfigurationTests(unittest.TestCase):
    def test_presets(self):
        s, r = PRESETS['standard'], PRESETS['reference']
        self.assertTrue(s.reg and s.init == 'clusters' and s.freeze_endmembers == 3000 and s.reg_floor == .005)
        self.assertTrue(not r.reg and r.init == 'fps' and r.freeze_endmembers == 300)

    def test_cli_overrides_and_regularization_switch(self):
        parse = lambda *a: train.config(cli.parse(['train', '--pack', 'p', '--output', 'o', *a]))
        self.assertEqual(parse(), PRESETS['standard'])
        self.assertEqual(parse('--reference'), PRESETS['reference'])
        self.assertEqual(parse('--reference', '-K', '16').materials, 16)
        off = parse('--no-regularization')
        self.assertEqual((off.reg, off.freeze_endmembers, off.init), (False, 300, 'clusters'))
        self.assertEqual(parse('--reg-floor', '0').reg_floor, 0.)

    def test_regularization_weight_schedule(self):
        cfg = Decoder()
        self.assertEqual(train.regularization_weight(0, cfg), .05)
        self.assertAlmostEqual(train.regularization_weight(1500, cfg), .025)
        self.assertEqual(train.regularization_weight(2950, cfg), .005)                  # floor reached before the fade ends
        self.assertEqual(train.regularization_weight(29999, cfg), .005)
        self.assertEqual(train.regularization_weight(29999, Decoder(reg_floor=0.)), 0.)
        self.assertEqual(train.regularization_weight(0, Decoder(reg=False)), 0.)

    def test_cosine_schedule(self):
        self.assertEqual(train.cosine_lr(0, 300), 1e-3)
        self.assertAlmostEqual(train.cosine_lr(299, 300), 1e-5)
        self.assertAlmostEqual(train.cosine_lr(1000, 300), 1e-5)            # clamped after the last step


class SceneConfigTests(unittest.TestCase):
    def test_every_scene_loads(self):
        self.assertGreaterEqual(len(scenes.names()), 22)
        local = scenes.LOCAL_ROOTS.exists()
        for name in scenes.names():
            config = scenes.load(name)
            with self.subTest(scene=name):
                self.assertIn(config.get('extends'), ('swr_bench', 'nespof', 'hsi_sc', 'hs_nerf'))
                self.assertIsInstance(scenes.get(config, 'decoder.materials'), int)
                self.assertIn(scenes.get(config, 'decoder.preset'), PRESETS)
                for option_map in scenes.OPTIONS.values():
                    for key in option_map.values():
                        scenes.get(config, key)
                if local:                                   # with this machine's roots, every recorded path exists
                    for section in ('data', 'packs', 'paper_model', 'pack_export'):
                        for key, value in (config.get(section) or {}).items():
                            if isinstance(value, str) and value.startswith('/'):
                                self.assertTrue(Path(value).exists(), f'{name}: {section}.{key} = {value}')

    def test_roots(self):
        with mock.patch.dict(os.environ, {'ABGS_DATA_ROOT': '/data'}):
            self.assertEqual(scenes.resolve('${ABGS_DATA_ROOT}/packs/x/train', {}), '/data/packs/x/train')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ABGS_MODELS_ROOT', None)
            local = {'links': {'${ABGS_MODELS_ROOT}/x': '/a/b', '${ABGS_MODELS_ROOT}/x/decoder': '/c'},
                     'roots': {'ABGS_MODELS_ROOT': '/m'}}
            self.assertEqual(scenes.resolve('${ABGS_MODELS_ROOT}/x/decoder/model_ema.pt', local), '/c/model_ema.pt')
            self.assertEqual(scenes.resolve('${ABGS_MODELS_ROOT}/x/source', local), '/a/b/source')
            self.assertEqual(scenes.resolve('${ABGS_MODELS_ROOT}/y', local), '/m/y')
            self.assertEqual(scenes.resolve('${ABGS_MODELS_ROOT}/y', {}), '${ABGS_MODELS_ROOT}/y')

    def test_command_line_overrides_scene_and_scene_overrides_defaults(self):
        args = cli.parse(['eval', '--model', 'm', '--pack', 'p', '--output', 'o'])
        self.assertEqual((args.foreground, args.clean_hsi_subdir, args.scorer), ('coverage', 'hsi', None))
        with mock.patch.dict(os.environ, {'ABGS_DATA_ROOT': '/d', 'ABGS_MODELS_ROOT': '/m'}):
            args = cli.parse(['train', '--scene', 'market', '--output', 'o'])
            self.assertEqual((args.materials, args.preset, str(args.clusters)), (24, 'standard', '/m/market/clusters_K24'))
            args = cli.parse(['train', '--scene', 'market', '--output', 'o', '-K', '16', '--reference'])
            self.assertEqual((args.materials, args.preset, args.clusters), (16, 'reference', None))   # K24 cache not for K16
            args = cli.parse(['eval', '--scene', 'nvidia_hq', '--output', 'o'])
            self.assertEqual((args.foreground, args.clean_hsi_subdir, str(args.pack)), ('surface', 'hsi', '/d/packs/nvidia_hq/eval'))
            args = cli.parse(['train', '--scene', 'nespof_scene1', '--output', 'o'])
            self.assertEqual((args.preset, args.materials, args.val_every_n_train), ('reference', 16, 5))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse(['eval', '--output', 'o'])

    def test_scorer_command_template(self):
        args = SimpleNamespace(scorer='benchmark', scorer_args=None, output=Path('/e'), dataset=None, clean_dataset=Path('/c'))
        pack = SimpleNamespace(meta={'source_path': '/noisy'})
        self.assertEqual(evaluate.scorer_command(args, pack)[1:],
                         ['-m', 'abundance_gs.scorers.benchmark', '--dataset', '/noisy', '--predictions', '/e/predictions',
                          '--endmembers', '/e/endmembers_radiance.npy', '--split', 'test', '--output',
                          '/e/benchmark_material_evaluation.json'])
        args.scorer, args.scorer_args = 'nvidia', scenes.get(scenes.load('nvidia_hq'), 'eval.scorer_args')
        self.assertEqual(evaluate.scorer_command(args, pack)[3:], ['--eval-dir', '/e', '--dataset', '/noisy', '--clean-dataset', '/c'])
        for name in scenes.names():
            scorer = scenes.get(scenes.load(name), 'eval.scorer')
            if scorer is not None:
                self.assertTrue((ROOT / 'abundance_gs/scorers' / f'{scorer}.py').is_file(), name)


class ModelTests(unittest.TestCase):
    def test_metric_values(self):
        g = torch.Generator().manual_seed(0)
        target = torch.rand(8, 6, 5, generator=g)
        result = metrics.unified_hsi_metrics(target.clone(), target, 1e-6)
        self.assertEqual((result['hsi_psnr'], result['hsi_mae']), (100.0, 0.0))
        noisy = metrics.unified_hsi_metrics((target + .1).clamp(0, 1), target, 1e-6)
        self.assertLess(noisy['hsi_psnr'], 30)

    def test_fast_mixing_after_compositing_equals_reference(self):
        torch.manual_seed(0)
        m = model.AbundanceField(torch.rand(40, 48) * 3 - 1.5, torch.rand(5, 9), torch.rand(9))
        with torch.no_grad():
            for p in m.network.parameters():
                p.add_(torch.randn_like(p) * .2)
        m.set_primitive_means(torch.randn(40, 3))
        counts = torch.randint(0, 6, (12,))
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
        total = int(offsets[-1])
        hits = {'offsets': offsets, 'ids': torch.randint(40, (total,), dtype=torch.int32), 'xyz': torch.rand(total, 3) - .5,
                'weight': torch.rand(total) * .3}
        source = SimpleNamespace(get=lambda i, keep=False: hits, pack=SimpleNamespace(height=3, width=4))
        center = torch.tensor([.3, -.2, 2.])
        ref = model.render_view(m, source, 0, center)
        with contextlib.redirect_stderr(io.StringIO()), mock.patch('warnings.warn'):
            fast = model.render_view_fast(m, source, 0, center, tau=0.)      # CPU: autocast off, exact up to summation order
        for a, b in zip(ref, fast):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_maps(self):
        torch.manual_seed(0)
        pack = SimpleNamespace(height=6, width=8)
        abundance, alpha, hsi = torch.rand(48, 5) - .1, torch.rand(48, 1), torch.rand(48, 7)
        self.assertTrue(torch.equal(inference.cube(hsi, pack), hsi.T.reshape(7, 6, 8)))
        for stride in (1, 2):
            maps = inference.abundance_npz(abundance, alpha, pack, stride)
            self.assertTrue(np.array_equal(maps['abundance'], abundance.T.reshape(-1, 6, 8)[:, ::stride, ::stride].clamp_min(0).half().numpy()))
            self.assertTrue(np.array_equal(maps['alpha'], alpha.reshape(6, 8)[::stride, ::stride].half().numpy()))

    def test_sampler_sequence(self):
        pack = SimpleNamespace(train=[0, 2, 3, 5, 7], pixels=1000)
        sampler, rng = train.Sampler(pack, SimpleNamespace(pixels=64, seed=42)), torch.Generator().manual_seed(42)
        for step in range(20):
            view, pix = sampler(step)
            self.assertEqual(view, pack.train[int(torch.randint(len(pack.train), (1,), generator=rng))])
            self.assertTrue(torch.equal(pix, torch.randint(1000, (64,), generator=rng)))


class DataAndClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory(prefix='abgs_package_test_')
        cls.root = Path(cls.tmp.name)
        cls.pack_path = fixture(cls.root / 'pack')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hsi_training_views(self):
        pack = data.Pack(self.pack_path)
        self.assertEqual(data.hsi_training_views(pack), list(range(8)))
        self.assertEqual(data.hsi_training_views(pack, 6), list(range(6)))
        self.assertEqual(data.hsi_training_views(pack, 0, 2), [0, 2, 4, 6])
        self.assertEqual(data.hsi_training_views(pack, 3, 5), [0, 1, 2])
        with self.assertRaises(ValueError):
            data.hsi_training_views(pack, -1)

    def test_prepare_validate_load(self):
        out = self.root / 'cache'
        with contextlib.redirect_stdout(io.StringIO()):
            clusters.prepare(self.pack_path, out, 3, train_indices=[0, 2, 4, 6])
        pack = data.Pack(self.pack_path)
        pack.train = [0, 2, 4, 6]
        audit = clusters.validate_cache(out, ['v00', 'v02', 'v04', 'v06'], 3, pack=pack)
        self.assertEqual(audit['materials'], 3)
        initial, labels, _ = clusters.load_pseudo_labels(out, pack, 3)
        self.assertEqual(initial.shape, (3, 21))
        self.assertTrue(torch.equal(labels[0].reshape(16, 16), torch.from_numpy(np.load(out / 'dense_labels/v00.npy')).long()))
        self.assertEqual(sorted(np.unique(labels[0].numpy())), [-1, 0, 1, 2])      # three clusters, background unassigned

    def test_older_cache_format_is_read(self):
        out = self.root / 'older_cache'
        with contextlib.redirect_stdout(io.StringIO()):
            clusters.prepare(self.pack_path, out, 3, train_indices=list(range(8)))
        record = json.loads((out / 'preparation.json').read_text())
        record['full_train_teacher_counts'] = record.pop('full_train_label_counts')
        (out / 'preparation.json').write_text(json.dumps(record))
        self.assertIn('full_train_label_counts', clusters.read_preparation(out))
        pack = data.Pack(self.pack_path)
        clusters.validate_cache(out, [f'v{i:02d}' for i in range(8)], 3, pack=pack)

    def test_farthest_point_initialization(self):
        pack = data.Pack(self.pack_path)
        initial = clusters.farthest_point_endmembers(pack, 3)
        self.assertEqual(initial.shape, (3, 21))
        torch.testing.assert_close(initial.norm(dim=1), torch.ones(3))

    def test_train_preflight_rejects_before_any_cuda_or_pack_read(self):
        out = self.root / 'never_created'
        base = ['train', '--pack', str(self.root / 'missing_pack'), '--output', str(out)]
        cases = [([], 'exactly one'), (['--source', str(self.root / 'missing.pt')], 'readable file'),
                 (['--source', str(self.pack_path / 'pack.json'), '--reg-floor', 'nan'], 'finite and nonnegative'),
                 (['--source', str(self.pack_path / 'pack.json'), '--hit-cache-gib', 'inf'], 'finite and nonnegative'),
                 (['--source', str(self.pack_path / 'pack.json'), '--hit-cache-gib', '16'], 'does not fit --gpu-memory-budget-mib 10500'),
                 (['--source', str(self.pack_path / 'pack.json'), '--steps', '0'], 'must be positive')]
        no_cuda = mock.patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('CUDA accessed'))
        with no_cuda, mock.patch.object(train, 'Pack', side_effect=AssertionError('pack read too early')):
            for extra, message in cases:
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                    train.run(cli.parse(base + extra))
        self.assertFalse(out.exists())

    def test_automatic_hit_cache_size(self):
        cfg = train.config(cli.parse(['train', '--pack', 'p', '--output', 'o']))
        self.assertEqual(cfg.hit_cache_gib, 'auto')
        gib = 2 ** 30
        with mock.patch.object(torch.cuda, 'mem_get_info', return_value=(22 * gib, 24 * gib)):
            self.assertEqual(train.gpu_memory(cfg, False), (20 * 1024, 17.))        # budget raised to free - margin
            self.assertEqual(train.gpu_memory(cfg, True), (10500, (10500 - 3072) / 1024))
        with mock.patch.object(torch.cuda, 'mem_get_info', return_value=(6 * gib, 24 * gib)):
            self.assertEqual(train.gpu_memory(cfg, False), (10500, 1.))             # busy card: the rest on the host
        fixed = train.config(cli.parse(['train', '--pack', 'p', '--output', 'o', '--hit-cache-gib', '4']))
        self.assertEqual(train.gpu_memory(fixed, False), (10500, 4.))

    def test_cached_K_mismatch_fails_before_cuda(self):
        cache = self.root / 'k3_cache'
        with contextlib.redirect_stdout(io.StringIO()):
            clusters.prepare(self.pack_path, cache, 3, train_indices=list(range(8)))
        argv = ['train', '--pack', str(self.pack_path), '--output', str(self.root / 'o'), '--source', str(self.pack_path / 'pack.json'),
                '--clusters', str(cache), '-K', '4']
        with mock.patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('CUDA accessed')):
            with self.assertRaisesRegex(ValueError, 'Cached K'):
                train.run(cli.parse(argv))


class RecordTests(unittest.TestCase):
    def test_record_round_trip_and_matching(self):
        with tempfile.TemporaryDirectory(prefix='abgs_package_test_') as tmp:
            out = Path(tmp)
            self.assertIsNone(records.read(out, 'source'))
            records.complete(out, 'source', {'pack': 'a'}, {'steps': 5}, checkpoint='ckpt.pt')
            record = records.read(out, 'source')
            self.assertEqual((record.checkpoint, record.schema_version), ('ckpt.pt', records.SCHEMA_VERSION))
            self.assertIsNone(records.read(out, 'train'))
            self.assertTrue(records.is_complete(out, 'source', {'pack': 'a'}, {'steps': 5}))
            self.assertFalse(records.is_complete(out, 'source', {'pack': 'b'}, {'steps': 5}))
            self.assertFalse(records.is_complete(out, 'source', {'pack': 'a'}, {'steps': 6}))

    def test_outputs_without_records_are_read_from_stage_files(self):
        with tempfile.TemporaryDirectory(prefix='abgs_package_test_') as tmp:
            out = Path(tmp)
            (out / 'status.json').write_text(json.dumps({'status': 'complete', 'metrics_ema': {'psnr': 30.}}))
            record = records.read(out, 'train')
            self.assertEqual((record.inputs, record.metrics_ema), (None, {'psnr': 30.}))
            self.assertTrue(records.is_complete(out, 'train', {'pack': 'x'}, {}))
            self.assertFalse(records.is_complete(out, 'anchor', {}, {}))


    def test_run_skips_stages_whose_records_match(self):
        with tempfile.TemporaryDirectory(prefix='abgs_package_test_') as tmp:
            tmp = Path(tmp)
            pack = fixture(tmp / 'pack')
            (tmp / 'scene').mkdir()
            (tmp / 'scene/image.png').write_bytes(b'x')
            calls = []

            def fake_stage(name, gpu, *argv):
                calls.append(name)
                a = cli.parse([name, *map(str, argv)])
                a.output.mkdir(parents=True, exist_ok=True)
                outputs = {}
                if name == 'source':
                    outputs['checkpoint'] = str(a.output / 'ckpt.pt')
                    Path(outputs['checkpoint']).write_bytes(b'ckpt')
                elif name == 'prepare':
                    (a.output / 'preparation.json').write_text('{"complete": true}')
                elif name == 'train':
                    outputs['model'] = str(a.output / 'model_ema.pt')
                records.complete(a.output, name, *records.expected(name, a), **outputs)

            def fake_anchor(a):
                calls.append('anchor')
                a.output.mkdir(parents=True)
                ply = a.output / 'point_cloud.ply'
                ply.write_bytes(b'ply')
                records.complete(a.output, 'anchor', *records.expected('anchor', a), ply=str(ply))

            argv = ['run', '--workdir', str(tmp / 'w'), '--scene-dir', str(tmp / 'scene'), '--resolution', '1',
                    '--pack', str(pack), '-K', '3']
            with mock.patch.object(pipeline, 'stage', fake_stage), mock.patch.object(pipeline, 'anchor_in_process', fake_anchor), \
                    contextlib.redirect_stdout(io.StringIO()):
                pipeline.run(cli.parse(argv), None)
                self.assertEqual(calls, ['anchor', 'source', 'prepare', 'train'])
                pipeline.run(cli.parse(argv), None)
                self.assertEqual(len(calls), 4)                    # every record matches: nothing reruns
                pipeline.run(cli.parse(argv + ['--seed', '7']), None)
                self.assertEqual(calls[4:], ['source', 'train'])   # the seed enters source and train only

class HitTests(unittest.TestCase):
    def test_host_resident_selection_equals_device_selection(self):
        g = torch.Generator().manual_seed(0)
        counts = torch.randint(0, 6, (50,), generator=g)
        total = int(counts.sum())
        hits = {'offsets': torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]),
                'ids': torch.randint(0, 1000, (total,), generator=g, dtype=torch.int32),
                'xyz': torch.randn(total, 3, generator=g), 'weight': torch.rand(total, generator=g)}
        resident = model.Hits({'means': torch.zeros(1, 3)}, None, 0)
        resident.host[0], resident.cache[1] = hits, hits
        pixels = torch.randint(0, 50, (64,), generator=g)
        (a, ida), (b, idb) = resident.select(0, pixels), resident.select(1, pixels)
        self.assertTrue(torch.equal(ida, idb))
        for k in ('ids', 'xyz', 'weight'):
            self.assertTrue(torch.equal(a[k], b[k]))


    def test_trace_error_is_not_masked_by_the_cleanup(self):
        def set_trace(mode, *tensors):
            if mode == 0:
                raise RuntimeError('trace shape mismatch')

        def rasterization(**kw):
            raise torch.OutOfMemoryError('CUDA out of memory')
        fake = (SimpleNamespace(set_nht_hit_trace=set_trace), rasterization, lambda **kw: None)
        splats = {k: torch.zeros(2, n) for k, n in (('means', 3), ('quats', 4), ('scales', 3), ('features', 48))}
        splats['opacities'] = torch.zeros(2)
        pack = SimpleNamespace(height=4, width=4, c2w=lambda i: torch.eye(4), K=lambda i: torch.eye(3))
        with mock.patch.object(model, '_gsplat', return_value=fake), self.assertRaisesRegex(torch.OutOfMemoryError, 'out of memory'):
            model.Hits(splats, pack, 0).trace(0)


@unittest.skipUnless(os.environ.get('ABGS_SMOKE_GPU'), 'set ABGS_SMOKE_GPU / _PACK / _SOURCE / _OUT for the GPU smoke test')
class GPUSmoke(unittest.TestCase):
    def test_fused_renderer_matches_fast(self):
        """Fused CUDA kernels (csrc/fused_render.cu) against render_view_fast on random hits of a random decoder."""
        if not torch.cuda.is_available():
            self.skipTest('needs a visible GPU (env/run.sh <gpu> ...)')
        g = torch.Generator().manual_seed(1)
        n, K, B = 4096, 24, 31
        field = model.AbundanceField(torch.randn(500, 48, generator=g), torch.rand(K, B, generator=g), torch.rand(B, generator=g))
        for p in field.network.parameters():
            p.data.normal_(0, .2, generator=g)
        field.camera_adapter.weight.data.normal_(0, .2, generator=g)
        field = field.cuda()
        field.set_primitive_means(torch.randn(500, 3).cuda())
        counts = torch.randint(0, 40, (n,), generator=g)
        total = int(counts.sum())
        hits = {'offsets': torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).cuda(),
                'ids': torch.randint(0, 500, (total,), generator=g, dtype=torch.int32).cuda(),
                'xyz': (torch.rand(total, 3, generator=g) - .5).cuda(),
                'weight': (torch.rand(total, generator=g) ** 4 * .3).cuda()}
        source = SimpleNamespace(pack=SimpleNamespace(height=64, width=64), get=lambda i, keep=False: hits)
        center = torch.tensor([3., -1., 2.]).cuda()
        fast = model.render_view_fast(field, source, 0, center)
        fused = model.render_view_fused(field, source, 0, center)
        for a, b in zip(fast, fused):
            self.assertLess(float((a - b).abs().max()), 5e-3 * max(1., float(a.abs().max())))
        self.assertLess(float((fast[2] - fused[2]).abs().max()), 1e-5)

    def test_train_eval_render_fps(self):
        gpu, pack, source = os.environ['ABGS_SMOKE_GPU'], os.environ['ABGS_SMOKE_PACK'], os.environ['ABGS_SMOKE_SOURCE']
        out = Path(os.environ['ABGS_SMOKE_OUT'])
        entry = ['python3', str(ROOT / 'abgs.py')]
        env = {k: v for k, v in os.environ.items() if k not in ('ABGS_ENV', 'ABGS_CWD', 'CUDA_VISIBLE_DEVICES')}
        extra = ['--clusters', os.environ['ABGS_SMOKE_CLUSTERS']] if os.environ.get('ABGS_SMOKE_CLUSTERS') else []
        run = lambda *a: subprocess.run([*entry, *a, '--gpu', gpu], env=env, check=True)
        run('train', '--pack', pack, '--source', source, '--steps', '20', '-K', os.environ.get('ABGS_SMOKE_K', '16'),
            '--output', str(out / 'decoder'), *extra)
        status = json.loads((out / 'decoder/status.json').read_text())
        self.assertTrue(np.isfinite(status['metrics_ema']['hsi_psnr']))
        model_path = out / 'decoder/model_ema.pt'
        run('eval', '--model', str(model_path), '--pack', pack, '--output', str(out / 'eval'), '--fast')
        fast = json.loads((out / 'eval/spectral_metrics.json').read_text())['summary']['unified_macro_mean']
        self.assertAlmostEqual(fast['hsi_psnr'], status['metrics_ema']['hsi_psnr'], delta=1e-2)
        first = data.Pack(pack).views[data.Pack(pack).test[0]]['name']
        run('render', '--model', str(model_path), '--pack', pack, '--output', str(out / 'render'), '--views', first, '--fast')
        run('fps', '--model', str(model_path), '--pack', pack, '--output', str(out / 'fps.json'))
        self.assertGreater(json.loads((out / 'fps.json').read_text())['fast']['fps'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

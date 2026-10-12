"""Stage `run`: every stage for one scene; stages whose output record (records.py) matches the current inputs and
options are skipped. The anchor runs in this process; the other stages run as subprocesses of the package entry point.

WORKDIR/pack/ (unless --pack)  ->  anchor/  ->  source/  ->  clusters_K<K>/  ->  decoder/  ->  eval/ (with --eval-pack)

A stage is chained with `ensure(name, gpu, *options)`, which needs the stage in cli.STAGES and its record in
records.STAGE_RECORDS; the record's output fields (e.g. `checkpoint`, `model`) feed the next stage.
"""
import json
import shlex
import subprocess
import sys
from pathlib import Path

from . import cli, records

ENTRY = Path(__file__).resolve().parents[1] / 'abgs.py'


def stage(name, gpu, *argv):
    command = [sys.executable, str(ENTRY), name, *map(str, argv)] + (['--gpu', gpu] if gpu is not None else [])
    print('[abgs run]', ' '.join(shlex.quote(c) for c in command), flush=True)
    subprocess.run(command, check=True)


def stage_args(name, *argv):
    return cli.parse([name, *map(str, argv)])


def complete(directory, name, args):
    """True if `directory` holds the output of stage `name` for these inputs and options (abundance_gs/records.py)."""
    return records.is_complete(directory, name, *records.expected(name, args))


def anchor_in_process(args):
    """Stage anchor in this process (FastGS trainer + scoring on the cameras it loaded); GPU memory released after."""
    import gc
    import torch
    from . import anchor
    anchor.run(args)
    gc.collect()
    torch.cuda.empty_cache()


def ensure_pack(args, w, gpu):
    if args.pack is not None:
        return args.pack
    if not (args.state and args.scene_id):
        raise ValueError('Give --pack, or --state and --scene-id to export one')
    pack = w / 'pack'
    if not (pack / 'pack.json').exists():
        stage('pack', gpu, '--state', args.state, '--scene-id', args.scene_id, '--output', pack)
    return pack


def eval_options(args):
    extra = ['--foreground', args.foreground] + (['--fast'] if args.fast else [])
    if args.clean_dataset is not None:
        extra += ['--clean-dataset', args.clean_dataset, '--clean-hsi-subdir', args.clean_hsi_subdir]
    if args.scorer is not None:
        extra += ['--scorer', args.scorer]
        if args.scorer_args is not None:
            scorer_args = args.scorer_args if isinstance(args.scorer_args, str) else shlex.join(args.scorer_args)
            extra += ['--scorer-args', scorer_args]
        if args.dataset is not None:
            extra += ['--dataset', args.dataset]
    return extra


def ensure(name, gpu, *argv):
    """Stage `name` with these options, run as a subprocess unless its output record already matches; its record."""
    a = stage_args(name, *argv)
    if not complete(a.output, name, a):
        stage(name, gpu, *argv)
    return records.read(a.output, name)


def run(args, gpu):
    w = args.workdir.resolve()
    w.mkdir(parents=True, exist_ok=True)
    pack = ensure_pack(args, w, gpu)
    a = stage_args('anchor', '--scene-dir', args.scene_dir, '--resolution', args.resolution, '--output', w / 'anchor')
    if not complete(a.output, 'anchor', a):
        anchor_in_process(a)
    ply = records.read(a.output, 'anchor').ply
    checkpoint = ensure('source', gpu, '--pack', pack, '--anchor-ply', ply, '--seed', args.seed, '--output', w / 'source').checkpoint
    subset = ['--train-max-views', args.train_max_views, '--val-every-n-train', args.val_every_n_train]
    cache = w / f'clusters_K{args.materials}'
    if args.preset == 'standard':                      # the reference preset reads no cluster cache
        ensure('prepare', None, '--pack', pack, '--materials', args.materials, '--output', cache, *subset)
    cluster_cache = ['--clusters', cache] if args.preset == 'standard' else []
    decoder = ensure('train', gpu, '--pack', pack, '--source', checkpoint, '--preset', args.preset, *cluster_cache,
                     '--materials', args.materials, '--seed', args.seed, '--output', w / 'decoder', *subset,
                     *shlex.split(args.train_args))
    if args.eval_pack is not None:
        ensure('eval', gpu, '--model', decoder.model, '--pack', args.eval_pack, '--output', w / 'eval', *eval_options(args))
    print(json.dumps({'workdir': str(w), 'model': decoder.model, 'metrics_ema': decoder.metrics_ema}, indent=1))

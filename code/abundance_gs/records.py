"""Output records of the stages: what a stage produced, from which inputs and with which configuration.

Every stage writes `abgs_record.json` into its output directory (a `StageRecord` subclass below, with a
schema_version). Stage `run` decides from these records whether a stage is already complete for the current inputs
and configuration. Output directories written before records existed are read through their stage files
(anchor.json, run.json, preparation.json, status.json, spectral_metrics.json); their inputs are unknown and are
accepted as they are.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 1
FILE = 'abgs_record.json'


@dataclass
class StageRecord:
    stage: str
    inputs: dict = field(default_factory=dict)         # name -> digest of the input file / directory
    config: dict = field(default_factory=dict)         # options that determine the output
    status: str = 'complete'
    schema_version: int = SCHEMA_VERSION


@dataclass
class AnchorRecord(StageRecord):
    ply: str = ''
    primitives: int = 0
    rgb_psnr_test: float = 0.


@dataclass
class SourceRecord(StageRecord):
    checkpoint: str = ''


@dataclass
class ClustersRecord(StageRecord):
    cache: str = ''


@dataclass
class DecoderRecord(StageRecord):
    model: str = ''
    metrics_ema: dict = field(default_factory=dict)


@dataclass
class EvalRecord(StageRecord):
    spectral_metrics: str = ''
    material_scores: str = ''


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path):
    """Digest of a directory's file list (relative path and size of every file)."""
    root = Path(path)
    entries = sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob('*') if p.is_file())
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def digest(path):
    path = Path(path)
    return tree_digest(path) if path.is_dir() else file_digest(path)


def write(directory, record):
    path = Path(directory) / FILE
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(asdict(record), indent=1) + '\n')
    temporary.replace(path)
    return path


def _from_stage_files(directory, stage):
    """Record of an output directory without abgs_record.json, from the stage's own files (inputs unknown)."""
    d = Path(directory)
    load = lambda name: json.loads((d / name).read_text()) if (d / name).exists() else None
    if stage == 'anchor' and (a := load('anchor.json')) and a.get('status') == 'complete':
        return AnchorRecord('anchor', inputs=None, ply=a['ply'], primitives=a['primitives'], rgb_psnr_test=a['rgb_psnr_test'])
    if stage == 'source' and (r := load('run.json')) and r.get('status') == 'complete':
        return SourceRecord('source', inputs=None, checkpoint=r['checkpoint'])
    if stage == 'prepare' and (p := load('preparation.json')) and p.get('complete'):
        return ClustersRecord('prepare', inputs=None, cache=str(d))
    if stage == 'train' and (s := load('status.json')) and s.get('status') == 'complete':
        return DecoderRecord('train', inputs=None, model=str(d / 'model_ema.pt'), metrics_ema=s['metrics_ema'])
    if stage == 'eval' and (d / 'spectral_metrics.json').exists():
        scores = d / 'benchmark_material_evaluation.json'
        return EvalRecord('eval', inputs=None, spectral_metrics=str(d / 'spectral_metrics.json'),
                          material_scores=str(scores) if scores.exists() else '')
    return None


def read(directory, stage):
    """The record of a stage output directory, or None if the stage has not completed there."""
    path = Path(directory) / FILE
    if path.exists():
        values = json.loads(path.read_text())
        kind = TYPES.get(values.get('stage'), StageRecord)
        record = kind(**{f.name: values[f.name] for f in fields(kind) if f.name in values})
        return record if record.stage == stage and record.status == 'complete' else None
    return _from_stage_files(directory, stage)


def is_complete(directory, stage, inputs, config):
    """True if `directory` holds a complete output of `stage` for these inputs and this configuration."""
    record = read(directory, stage)
    if record is None:
        return False
    if record.inputs is None:                          # output written before records: accepted as it is
        return True
    return record.inputs == inputs and record.config == config


def complete(directory, stage, inputs, config, **outputs):
    """Write the record of a completed stage."""
    return write(directory, TYPES[stage](stage, inputs=inputs, config=config, **outputs))


def _path(value):
    return None if value is None else str(value)


def _anchor_inputs(args):
    return ({'scene_dir': digest(args.scene_dir)},
            {'resolution': str(args.resolution), 'iterations': args.iterations, 'seed': args.seed})


def _source_inputs(args):
    pack = Path(args.pack) / 'pack.json'
    ply = args.anchor_ply or json.loads(pack.read_text()).get('anchor_ply')
    return {'pack': digest(pack), 'anchor_ply': digest(ply)}, {'steps': args.steps, 'seed': args.seed}


def _prepare_inputs(args):
    return ({'pack': digest(Path(args.pack) / 'pack.json')},
            {'materials': args.materials, 'train_max_views': args.train_max_views,
             'val_every_n_train': args.val_every_n_train, 'device': args.device})


def _train_inputs(args):
    from .train import config, uses_clusters
    clusters = None
    if uses_clusters(config(args)):                    # the cache train reads (its default: OUTPUT/appearance_clusters)
        clusters = Path(args.clusters if args.clusters is not None else Path(args.output) / 'appearance_clusters') / 'preparation.json'
    inputs = {'pack': digest(Path(args.pack) / 'pack.json'),
              'source': digest(args.source) if args.source is not None else None,
              'anchor_ply': digest(args.anchor_ply) if args.anchor_ply is not None else None,
              'clusters': digest(clusters) if clusters is not None and clusters.exists() else None}
    names = ('preset', 'materials', 'steps', 'pixels', 'seed', 'init', 'reg', 'reg_floor', 'freeze_endmembers', 'w_sam',
             'train_max_views', 'val_every_n_train')
    return inputs, {k: getattr(args, k) for k in names}


def _eval_inputs(args):
    names = ('clean_dataset', 'clean_hsi_subdir', 'foreground', 'scorer', 'scorer_args', 'dataset', 'fast', 'fused', 'class_spectral')
    return ({'model': digest(args.model), 'pack': digest(Path(args.pack) / 'pack.json')},
            {k: _path(v) if k in ('clean_dataset', 'dataset') else v for k, v in ((k, getattr(args, k)) for k in names)})


# stage -> (record type, (inputs, config) of an invocation); a new stage with a record adds one entry here
STAGE_RECORDS = {'anchor': (AnchorRecord, _anchor_inputs), 'source': (SourceRecord, _source_inputs),
                 'prepare': (ClustersRecord, _prepare_inputs), 'train': (DecoderRecord, _train_inputs),
                 'eval': (EvalRecord, _eval_inputs)}
TYPES = {stage: kind for stage, (kind, _) in STAGE_RECORDS.items()}


def expected(stage, args):
    """(inputs, config) of a stage invocation: what its record must match for the output to be reused."""
    if stage not in STAGE_RECORDS:
        raise ValueError(f'No record for stage {stage!r}')
    return STAGE_RECORDS[stage][1](args)

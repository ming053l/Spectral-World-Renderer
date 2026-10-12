"""Scene configurations (standard library only): configs/scenes/<name>.json on top of configs/datasets/<extends>.json.

A scene file records what the paper used for that scene: data directories, packs, K and preset, evaluation
settings, views, and the finished models (`paper_model`). With `--scene NAME` every stage option listed in OPTIONS
below that is not given on the command line is taken from the merged configuration.

Paths are written relative to two roots, ${ABGS_DATA_ROOT} (datasets, packs) and ${ABGS_MODELS_ROOT} (trained
models), taken from the environment. Without these variables, the untracked file configs/local_roots.json resolves
them on a given machine: "links" maps logical path prefixes to existing locations (longest prefix wins) and "roots"
gives a value per variable. Stage `scenes` lists the configurations.
"""
import json
import os
import re
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
LOCAL_ROOTS = CONFIGS / 'local_roots.json'
_VARIABLE = re.compile(r'\$\{(\w+)\}')

_SUBSET = {'train_max_views': 'decoder.train_max_views', 'val_every_n_train': 'decoder.val_every_n_train'}
_EVAL = {'clean_dataset': 'data.clean_dataset', 'clean_hsi_subdir': 'data.clean_hsi_subdir', 'dataset': 'data.noisy_dataset',
         'scorer': 'eval.scorer', 'scorer_args': 'eval.scorer_args', 'foreground': 'eval.foreground'}
# stage -> {option (argparse dest): dotted key of the merged configuration}
OPTIONS = {
    'anchor': {'scene_dir': 'data.anchor_scene', 'resolution': 'anchor.resolution'},
    'pack': {'state': 'pack_export.state', 'scene_id': 'scene_id', 'source_path': 'pack_export.source_path',
             'anchor': 'pack_export.anchor'},
    'source': {'pack': 'packs.train', 'anchor_ply': 'paper_model.anchor_ply'},
    'prepare': {'pack': 'packs.train', 'materials': 'decoder.materials', **_SUBSET},
    'train': {'pack': 'packs.train', 'source': 'paper_model.source', 'clusters': 'paper_model.clusters',
              'preset': 'decoder.preset', 'materials': 'decoder.materials', **_SUBSET},
    'eval': {'model': 'paper_model.model', 'pack': 'packs.eval', 'retain_views': 'views.retain', **_EVAL},
    'render': {'model': 'paper_model.model', 'pack': 'packs.eval', 'figure_views': 'views.figures'},
    'fps': {'model': 'paper_model.model', 'pack': 'packs.eval'},
    'run': {'scene_dir': 'data.anchor_scene', 'resolution': 'anchor.resolution', 'pack': 'packs.train',
            'state': 'pack_export.state', 'scene_id': 'scene_id', 'preset': 'decoder.preset',
            'materials': 'decoder.materials', 'eval_pack': 'packs.eval', **_SUBSET, **_EVAL},
}
# filled only when the configuration value belongs to what the run will use
_SKIP_IF_GIVEN = {'source': 'anchor_ply'}              # --anchor-ply replaces the source field


def names():
    return sorted(p.stem for p in (CONFIGS / 'scenes').glob('*.json'))


def _merge(base, over):
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _local():
    return json.loads(LOCAL_ROOTS.read_text()) if LOCAL_ROOTS.exists() else {}


def resolve(value, local=None):
    """A configuration path with its ${ROOT} variables resolved (unchanged where a variable is not set anywhere)."""
    if not isinstance(value, str) or '${' not in value:
        return value
    local = _local() if local is None else local
    if not any(variable in os.environ for variable in _VARIABLE.findall(value)):
        links = local.get('links', {})
        prefix = max((k for k in links if value == k or value.startswith(k + '/')), key=len, default=None)
        if prefix is not None:
            return links[prefix] + value[len(prefix):]
    roots = local.get('roots', {})
    return _VARIABLE.sub(lambda m: os.environ.get(m.group(1)) or roots.get(m.group(1)) or m.group(0), value)


def _resolve_all(value, local):
    if isinstance(value, dict):
        return {k: _resolve_all(v, local) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_all(v, local) for v in value]
    return resolve(value, local)


def load(name):
    """Merged configuration of a scene (its dataset file first, the scene file on top), paths resolved."""
    path = CONFIGS / 'scenes' / f'{name}.json'
    if not path.exists():
        raise SystemExit(f'Unknown scene {name!r}; known: {", ".join(names())}')
    scene = json.loads(path.read_text())
    base = {}
    if 'extends' in scene:
        base = json.loads((CONFIGS / 'datasets' / f'{scene["extends"]}.json').read_text())
        base['dataset_description'] = base.pop('description', None)
    return _resolve_all(_merge(base, scene), _local())


def get(config, dotted):
    value = config
    for key in dotted.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def fill(args, stage, config):
    """Set every option of `stage` that is still None from the scene configuration; returns {option: value} filled."""
    filled = {}
    for option, key in OPTIONS.get(stage, {}).items():
        value = get(config, key)
        if value is None or getattr(args, option, None) is not None:
            continue
        if option in _SKIP_IF_GIVEN and getattr(args, _SKIP_IF_GIVEN[option], None) is not None:
            continue
        filled[option] = value
    # a cluster cache is K-specific: only the paper's cache for the paper's K
    if 'clusters' in filled and args.materials is not None and args.materials != get(config, 'decoder.materials'):
        del filled['clusters']
    for option, value in filled.items():
        if isinstance(value, str) and '${' in value:
            raise SystemExit(f'{option}: {value} has an unset root; set ' + ', '.join(_VARIABLE.findall(value))
                             + ' (or map it in configs/local_roots.json)')
        setattr(args, option, value)
    return filled


def run(args):
    """Stage `scenes`: one line per configuration."""
    print(f'{"scene":16s} {"dataset":10s} {"K":>3s} {"preset":9s} paper model')
    for name in names():
        c = load(name)
        model = get(c, 'paper_model.model')
        print(f'{name:16s} {c.get("extends", "-"):10s} {get(c, "decoder.materials") or "-":>3} '
              f'{get(c, "decoder.preset") or "-":9s} {"yes" if model and Path(model).exists() else "no"}')
    print(f'\n{len(names())} scenes in {CONFIGS / "scenes"};  abgs <stage> --scene NAME --print-config')

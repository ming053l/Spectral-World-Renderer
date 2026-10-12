"""Command-line options of every stage (standard library only, so `--help` works in any python).

STAGES is the stage registry: module (imported only when the stage runs, entry point `run(args)`), environment, GPU
memory, description and the function that adds the stage's options. `parse` resolves the options in the order:
command line > --scene configuration (scenes.py) > built-in default.
"""
import argparse
import json
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from . import scenes
from .defaults import GPU_MARGIN_GIB, PRESETS, Anchor, Render, Source
from .scorers import SCORERS

FAST = f'fp16 renderer skipping hits of weight < {Render.fast_tau}'
FUSED = 'fast renderer with the fused CUDA kernels (csrc/fused_render.cu); agrees with --fast to fp16 rounding'


def _subset(ap):
    ap.add_argument('--train-max-views', type=int, default=0, help='only the first N TRAIN views give HSI supervision')
    ap.add_argument('--val-every-n-train', type=int, default=0, help='every n-th HSI TRAIN view is held out (validation)')


def _benchmark(ap):
    ap.add_argument('--clean-dataset', type=Path, help='benchmark layout: paired clean dataset (<subdir>/<view>.npy references)')
    ap.add_argument('--clean-hsi-subdir', default='hsi')
    ap.add_argument('--foreground', choices=['coverage', 'surface'], default='coverage',
                    help="GT foreground: 'coverage' = coverage > 0 (else material_id below the light-source class); "
                         "'surface' = the released surface_evaluation_mask (NVIDIA HQ)")
    ap.add_argument('--scorer', choices=SCORERS,
                    help='material scorer (abundance_gs/scorers), run after rendering')
    ap.add_argument('--scorer-args', help='scorer command line, placeholders {output} {predictions} {endmembers} {dataset} '
                                          '{clean_dataset} {report} (default: --dataset --predictions --endmembers --split test --output)')
    ap.add_argument('--dataset', type=Path, help='dataset given to the scorer (default: the pack source_path)')


def _model_pack_output(ap, output_help=None):
    ap.add_argument('--model', type=Path, required=True, help='model_ema.pt from stage `train`')
    ap.add_argument('--pack', type=Path, required=True, help='pack to render (e.g. pack_native: native resolution and intrinsics)')
    ap.add_argument('--output', type=Path, required=True, help=output_help)


def _anchor(ap):
    d = Anchor()
    ap.add_argument('--scene-dir', required=True, help='scene source directory (images/, sparse/ or transforms, hsi/)')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resolution', required=True, help='-r of the trainer: 1 on SWR-Bench half-resolution data / HS-NeRF, 512 NeSpoF / apple')
    ap.add_argument('--iterations', type=int, default=d.iterations)
    ap.add_argument('--seed', type=int, default=d.seed)


def _pack(ap):
    ap.add_argument('--state', required=True, help='reference state file (.pt with an \'args\' dict that defines the scene)')
    ap.add_argument('--scene-id', required=True)
    ap.add_argument('--output', type=Path, required=True, help='pack directory to write')
    ap.add_argument('--anchor', help='anchor model dir for the loader (default: model_path of the saved args)')
    ap.add_argument('--source-path', help='override of the saved source_path (data moved since the run)')
    ap.add_argument('--hsi-dir')


def _source(ap):
    d = Source()
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--anchor-ply', type=Path, help='anchor ply of this repeat (default: the one recorded in the pack)')
    ap.add_argument('--steps', type=int, default=d.steps)
    ap.add_argument('--seed', type=int, default=d.seed)


def _prepare(ap):
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='cache directory (created; an existing one is only validated)')
    ap.add_argument('--materials', '-K', type=int, required=True)
    _subset(ap)
    ap.add_argument('--check-only', action='store_true', help='validate an existing cache, never create one')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                    help='clustering device; cuda is faster, recorded in preparation.json (see README)')


def _gib_or_auto(value):
    return 'auto' if value == 'auto' else float(value)


def _train(ap):
    s, r = PRESETS['standard'], PRESETS['reference']

    def d(name):
        a, b = getattr(s, name), getattr(r, name)
        return f'(standard {a}; reference {b})' if a != b else f'(default {a})'
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--source', type=Path, help='source-field checkpoint from stage `source` (ckpts/ckpt_4999_rank0.pt)')
    g.add_argument('--anchor-ply', type=Path, help='no RGB source stage: dilated anchor geometry + random features go '
                                                   'straight into the decoder (diagnostic)')
    ap.add_argument('--preset', choices=sorted(PRESETS), default='standard')
    ap.add_argument('--reference', dest='preset', action='store_const', const='reference', help='= --preset reference (no appearance regularization)')
    ap.add_argument('--materials', '-K', type=int, help='number of endmembers K ' + d('materials'))
    ap.add_argument('--steps', type=int, help=d('steps'))
    ap.add_argument('--pixels', type=int, help='pixels per update ' + d('pixels'))
    ap.add_argument('--seed', type=int, help=d('seed'))
    ap.add_argument('--clusters', type=Path,
                    help='appearance-cluster cache from stage `prepare` (absent: built at OUTPUT/appearance_clusters)')
    ap.add_argument('--init', choices=['clusters', 'fps'], help='endmember initialization ' + d('init'))
    ap.add_argument('--no-regularization', dest='reg', action='store_false', default=None,
                    help='disable the persistent assignment regularization L_p (reference preset: off)')
    ap.add_argument('--reg-floor', type=float, help='floor of the L_p weight; 0 = fading-only ablation ' + d('reg_floor'))
    ap.add_argument('--freeze-endmembers', type=int, help='updates with a frozen dictionary (default 3000 with L_p, 300 without)')
    ap.add_argument('--w-sam', type=float, help=d('w_sam'))
    _subset(ap)
    ap.add_argument('--score-val', action='store_true', help='also score the validation views held out by --val-every-n-train')
    ap.add_argument('--eval-steps', default='', help='comma list of steps at which TEST is scored with EMA weights (curve.json)')
    ap.add_argument('--keep-cubes', action='store_true', help='save EMA TEST spectra and abundance maps under predictions/')
    ap.add_argument('--hit-cache-gib', type=_gib_or_auto,
                    help="GiB of the TRAIN hit lists (all traced once) kept on the GPU, the rest in pinned host memory; "
                         "'auto': free GPU memory minus a 2 GiB margin and a 3 GiB working set " + d('hit_cache_gib'))
    ap.add_argument('--gpu-memory-budget-mib', type=int, help=d('gpu_memory_budget_mib'))


def _eval(ap):
    _model_pack_output(ap)
    _benchmark(ap)
    ap.add_argument('--class-spectral', action='store_true', help='per-GT-class pooled TEST metrics (Karst)')
    ap.add_argument('--fast', action='store_true', help=FAST)
    ap.add_argument('--fused', action='store_true', help=FUSED)
    ap.add_argument('--retain-views', help='after the scorer: delete every abundance map except these views (comma list)')
    ap.add_argument('--gpu-memory-budget-mib', type=int, default=PRESETS['standard'].gpu_memory_budget_mib)


def _render(ap):
    _model_pack_output(ap)
    ap.add_argument('--views', default='test', help="'test', 'train', 'all', 'figures' (the scene's figure views) or a comma list of view names")
    ap.add_argument('--fast', action='store_true', help=FAST)
    ap.add_argument('--fused', action='store_true', help=FUSED)
    ap.add_argument('--radiance', action='store_true', help="multiply spectra by the pack's hsi_normalization_scale")
    ap.add_argument('--stride', type=int, default=1, help='spatial subsampling of the written maps')
    ap.add_argument('--no-spectra', action='store_true', help='write abundance maps only')
    ap.set_defaults(figure_views=None)


def _fps(ap):
    _model_pack_output(ap, 'json file')
    ap.add_argument('--mode', choices=['fast', 'reference', 'fused', 'both', 'all'], default='both',
                    help='both: reference and fast; all: reference, fast and fused')
    ap.add_argument('--tau', type=float, default=Render.fast_tau)


def _run(ap):
    ap.add_argument('--workdir', type=Path, required=True, help='anchor/ source/ clusters_K<K>/ decoder/ [eval/] are created here')
    g = ap.add_argument_group('scene')
    g.add_argument('--scene-dir', required=True, help='scene source directory for the anchor trainer')
    g.add_argument('--resolution', required=True, help='-r of the anchor trainer')
    g.add_argument('--pack', type=Path, help='existing pack (otherwise exported to WORKDIR/pack from --state)')
    g.add_argument('--state', help='reference state file for the pack export')
    g.add_argument('--scene-id')
    g = ap.add_argument_group('decoder')
    g.add_argument('--preset', choices=sorted(PRESETS), default='standard')
    g.add_argument('--materials', '-K', type=int, default=PRESETS['standard'].materials)
    g.add_argument('--seed', type=int, default=PRESETS['standard'].seed, help='source-field and decoder seed')
    _subset(g)
    g.add_argument('--train-args', default='', help='extra options for stage train, as one quoted string')
    g = ap.add_argument_group('evaluation (optional)')
    g.add_argument('--eval-pack', type=Path, help='pack to evaluate on (e.g. native resolution); omit to skip')
    _benchmark(g)
    g.add_argument('--fast', action='store_true')


def _build(ap):
    ap.add_argument('extensions', nargs='*', help='gsplat_cuda fastgs_rasterizer simple_knn fused_ssim_cuda gs_rasterizer abgs_render (default: all)')
    ap.add_argument('--adopt-from', help='library (or directory with <name>.so) built elsewhere from identical sources (copied, not rebuilt)')
    ap.add_argument('--old-root', action='append', default=[], help='source roots of that build, searched in order')


class Stage(NamedTuple):
    module: str                          # abundance_gs.<module>.run(args) implements the stage
    env: Optional[str]                   # environment the stage runs in (None: any python)
    gpu_mib: int                         # GPU memory needed (0 = CPU)
    description: str
    options: Callable[[argparse.ArgumentParser], None]


STAGES = {
    'scenes': Stage('scenes', None, 0, 'list the scene configurations (configs/scenes)', lambda ap: None),
    'anchor': Stage('anchor', 'abgs', 8000, 'RGB 3DGS anchor with multi-view density control (FastGS, 30k iterations)', _anchor),
    'pack': Stage('export', 'abgs', 0, 'export cameras / split / RGB / HSI paths of a scene into a pack', _pack),
    'source': Stage('source', 'abgs', 6000, 'anchor Gaussians -> harmonic-texture feature field, 5k-step RGB feature refinement',
                    _source),
    'prepare': Stage('clusters', 'abgs', 4000,
                     'appearance clusters: initial dictionary + pseudo-labels (GPU only with --device cuda)', _prepare),
    'train': Stage('train', 'abgs', 10500,
                   'train the abundance decoder (standard; --reference without the appearance regularization)', _train),
    'eval': Stage('evaluate', 'abgs', 10500, 'render a trained model on a (native) pack and score it', _eval),
    'render': Stage('render', 'abgs', 8000, 'write spectra and abundance maps of chosen views', _render),
    'fps': Stage('fps', 'abgs', 8000, 'render speed over the TEST views (reference and fast renderer)', _fps),
    'run': Stage('pipeline', 'abgs', 10500, 'all stages for one scene: [pack] anchor source prepare train [eval]', _run),
    'build': Stage('ext', 'abgs', 0, 'build (or adopt) the CUDA extensions into the build directory', _build),
}


class _Formatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        if action.default is None or action.default is False or action.help is None:
            return action.help
        return super()._get_help_string(action)


def _scene_fillable(parser, stage):
    """Options a scene can fill: argparse default None and not required; their default / requirement is kept here."""
    fallback, required = {}, []
    for action in parser._actions:
        if action.dest not in scenes.OPTIONS.get(stage, {}):
            continue
        if action.required:
            action.required = False
            required.append(action.dest)
            action.help = ((action.help or '') + ' (required unless given by --scene)').strip()
        elif action.default is not None:
            fallback[action.dest] = action.default
            action.help = ((action.help or '') + f' (default: {action.default})').strip()
            action.default = None
    parser.set_defaults(_fallback=fallback, _required=required)


def build_parser():
    ap = argparse.ArgumentParser(prog='abgs', description='Abundance-GS: one entry point for every stage. '
                                 'Every stage runs in the package environment (env/env.sh); see README.md.')
    sub = ap.add_subparsers(dest='stage', required=True, metavar='stage')
    for name, stage in STAGES.items():
        p = sub.add_parser(name, help=stage.description, description=stage.description, formatter_class=_Formatter)
        stage.options(p)
        _scene_fillable(p, name)
        for action in p._actions:
            if action.help is None and action.default not in (None, False, '', argparse.SUPPRESS):
                action.help = 'default: %(default)s'
        if name != 'scenes':
            g = p.add_argument_group('scene configuration')
            g.add_argument('--scene', metavar='NAME', help='fill unset options from configs/scenes/NAME.json (list: `abgs.py scenes`)')
            g.add_argument('--print-config', action='store_true', help='print the resolved options and exit')
        g = p.add_argument_group('environment')
        if stage.gpu_mib:
            g.add_argument('--gpu', default='auto', help=f'GPU index, or auto = the card with most free memory '
                                                          f'(>= {stage.gpu_mib} MiB + {GPU_MARGIN_GIB:g} GiB margin)')
        if stage.env:
            g.add_argument('--no-env', action='store_true', help='run in the current interpreter (environment already set up)')
    return ap


def parse(argv):
    """Options of one stage: command line, then --scene, then built-in defaults; missing required options are errors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    sub = parser._subparsers._group_actions[0].choices[args.stage]
    types = {a.dest: a.type for a in sub._actions}
    args.scene_filled = {}
    if getattr(args, 'scene', None):
        args.scene_filled = scenes.fill(args, args.stage, scenes.load(args.scene))
        for option, value in args.scene_filled.items():
            if isinstance(value, (str, int, float)) and types.get(option) is not None:
                setattr(args, option, types[option](value))
    for option, value in args._fallback.items():
        if getattr(args, option) is None:
            setattr(args, option, value)
    missing = [o for o in args._required if getattr(args, o) is None]
    if missing:
        sub.error('missing ' + ', '.join('--' + o.replace('_', '-') for o in missing) +
                  (f' (not in scene {args.scene!r})' if getattr(args, 'scene', None) else ''))
    del args._fallback, args._required
    return args


def gpu_need(args):
    """GPU MiB a stage needs (0 = runs on the CPU)."""
    if args.stage == 'prepare' and args.device == 'cpu':
        return 0
    return getattr(args, 'gpu_memory_budget_mib', None) or STAGES[args.stage].gpu_mib


def describe(args):
    """The resolved options as JSON (for --print-config)."""
    hidden = {'print_config', 'scene_filled'}
    values = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in hidden}
    return json.dumps({'options': values, 'from_scene': sorted(args.scene_filled)}, indent=1)

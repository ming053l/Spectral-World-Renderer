"""Command-line entry point (`abgs` console script and abgs.py): option parsing, GPU choice, re-execution in the
package environment (env/env.sh), stage dispatch.

    abgs <stage> [options]                  stages: scenes build anchor pack source prepare train eval render fps run
    abgs <stage> --help
    abgs <stage> --scene NAME [options]     options not given are taken from configs/scenes/NAME.json

The stage runs on --gpu (default: the card with most free memory, keeping a 2 GiB margin). Relative paths are
resolved against the directory the command was started from.
"""
import os
import subprocess
import sys
from pathlib import Path

from .cli import STAGES, describe, gpu_need, parse
from .defaults import GPU_MARGIN_GIB

HERE = Path(__file__).resolve().parents[1]           # folder holding env/ and abgs.py
MARGIN_MIB = int(GPU_MARGIN_GIB * 1024)


def pick_gpu(need_mib):
    """Index of the GPU with the most free memory, if it has need + margin free. Never touches other processes."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    free = sorted(((int(f), i.strip()) for i, f in (line.split(',') for line in out.strip().splitlines())), reverse=True)
    if not free or free[0][0] < need_mib + MARGIN_MIB:
        raise SystemExit(f'No GPU has {need_mib} MiB + {MARGIN_MIB} MiB margin free (free MiB per GPU: '
                         f'{ {i: f for f, i in free} }); pass --gpu explicitly or wait')
    return free[0][1]


def relaunch(gpu, argv):
    """Re-execute this command inside the package environment (env/env.sh)."""
    variables = dict(os.environ, ABGS_ENV='abgs', ABGS_CWD=os.getcwd())
    command = ['bash', '-c', 'set -e; source "$1"; export CUDA_VISIBLE_DEVICES="$2"; shift 2; exec "$ABGS_PYTHON" "$@"',
               'abgs', str(HERE / 'env/env.sh'), gpu, str(HERE / 'abgs.py'), *argv]
    os.execvpe(command[0], command, variables)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if os.environ.get('ABGS_CWD'):
        os.chdir(os.environ['ABGS_CWD'])
    args = parse(argv)
    if getattr(args, 'print_config', False):
        print(describe(args))
        return
    stage = STAGES[args.stage]
    module, env = stage.module, stage.env
    need = gpu_need(args)
    inside = env is not None and (os.environ.get('ABGS_ENV') == env or args.no_env)
    gpu = None
    if need and not (inside and os.environ.get('ABGS_ENV')):
        gpu = pick_gpu(need) if args.gpu == 'auto' else args.gpu
    if env is not None and not inside:
        relaunch(gpu if need else '', argv)
    if inside and gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if args.stage == 'prepare':                     # 2 BLAS threads for the clustering, set before numpy
        for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[variable] = '2'
    import importlib
    implementation = importlib.import_module('abundance_gs.' + module)
    if args.stage == 'run':
        implementation.run(args, gpu)
    else:
        implementation.run(args)


if __name__ == '__main__':
    main()

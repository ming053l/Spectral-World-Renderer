"""The package's CUDA extensions: built from abundance_gs/third_party with torch.utils.cpp_extension into paths.BUILD_DIR.

Each extension directory holds <name>.so and SOURCES.sha256, the digest of every source and header it was built
from (plus the compiler flags). A matching digest means the library is reused; otherwise it is (re)built. `adopt`
copies a library built elsewhere from byte-identical sources (checked file by file) instead of rebuilding it.

  gsplat_cuda        gsplat-NHT with the hit-trace kernel: source field (stage source), hit lists (train/eval/render/fps)
  fastgs_rasterizer  FastGS rasterizer (stage anchor)
  simple_knn         nearest-neighbour scales of the initial points (stage anchor)
  fused_ssim_cuda    fused SSIM loss (stages anchor and source)
  gs_rasterizer      3DGS rasterizer of the held-out RGB score (stage anchor)
  abgs_render        fused per-hit encode / compositing kernels of the fast renderer (abundance_gs/csrc, opt-in --fused)
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from . import paths

THIRD_PARTY = Path(__file__).resolve().parent / 'third_party'
_GSPLAT = THIRD_PARTY / 'gsplat' / 'cuda'
_HEADERS = ('.h', '.cuh', '.hpp', '.inl')
_GSPLAT_FLAGS = ['-std=c++20', '-O3', '-DNDEBUG', '-Wno-attributes', '-Wno-unused-but-set-variable', '-Wno-unknown-pragmas',
                 '-DAT_PARALLEL_OPENMP', '-fopenmp']


def _gsplat():
    sources = sorted(str(p) for p in (_GSPLAT / 'csrc').glob('*.cu')) + sorted(str(p) for p in (_GSPLAT / 'csrc').glob('*.cpp'))
    cuda_home = os.environ.get('CUDA_HOME', '')
    targets = [str(p) for p in Path(cuda_home).glob('targets/*/include')] if cuda_home else []
    return dict(root=_GSPLAT, sources=sources + [str(_GSPLAT / 'ext.cpp')],
                headers=[_GSPLAT / 'include', _GSPLAT / 'csrc', _GSPLAT / 'csrc/third_party/glm/glm'],
                include=[str(_GSPLAT / 'include'), str(_GSPLAT / 'csrc/third_party/glm')] + targets,
                cflags=list(_GSPLAT_FLAGS),
                cuda_cflags=['--forward-unknown-opts', '--split-compile', os.environ.get('GSPLAT_NVCC_SPLIT_COMPILE', '8'),
                             '-use_fast_math', '-diag-suppress', '3189,20012,186', '--expt-relaxed-constexpr'] + _GSPLAT_FLAGS,
                ldflags=['-s'])


def _inria(directory, sources, glm=False, cflags=(), cuda_cflags=()):
    root = THIRD_PARTY / directory
    include = [str(root / 'third_party/glm')] if glm else []
    return dict(root=root, sources=[str(root / s) for s in sources],
                headers=[root, root / 'cuda_rasterizer'] + ([root / 'third_party/glm/glm'] if glm else []),
                include=[], cflags=list(cflags), cuda_cflags=['-I' + i for i in include] + list(cuda_cflags), ldflags=[])


CSRC = Path(__file__).resolve().parent / 'csrc'


def _render():
    return dict(root=CSRC, sources=[str(CSRC / 'fused_render.cu')], headers=[CSRC], include=[], cflags=['-O3'],
                cuda_cflags=['-O3'], ldflags=[])


SPECS = {
    'gsplat_cuda': _gsplat,
    'fastgs_rasterizer': lambda: _inria('fastgs_rasterizer', ['cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/forward.cu',
                                                              'cuda_rasterizer/backward.cu', 'cuda_rasterizer/adam.cu',
                                                              'rasterize_points.cu', 'ext.cpp'], glm=True),
    'simple_knn': lambda: _inria('simple_knn', ['spatial.cu', 'simple_knn.cu', 'ext.cpp']),
    'fused_ssim_cuda': lambda: _inria('fused_ssim', ['ssim.cu', 'ssim3d.cu', 'ext.cpp'], cflags=['-O3', '-DFUSED_SSIM_CUDA'],
                                      cuda_cflags=['-O3', '-DFUSED_SSIM_CUDA', '--maxrregcount=32', '--use_fast_math']),
    'gs_rasterizer': lambda: _inria('gs_rasterizer', ['cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/forward.cu',
                                                      'cuda_rasterizer/backward.cu', 'rasterize_points.cu', 'ext.cpp'], glm=True),
    'abgs_render': _render,
}
_LOADED = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _files(spec):
    """{path relative to the extension root: absolute path} of every source and header."""
    root = spec['root']
    files = {str(Path(s).relative_to(root)): Path(s) for s in spec['sources']}
    for directory in spec['headers']:
        recursive = directory.name == 'glm' or directory.name == 'include'
        for p in (directory.rglob('*') if recursive else directory.glob('*')):
            if p.suffix in _HEADERS and p.is_file():
                files[str(p.relative_to(root))] = p
    return files


def digest(name):
    spec = SPECS[name]()
    files = _files(spec)
    record = {'files': {k: _sha(v) for k, v in sorted(files.items())},
              'flags': [spec['cflags'], spec['cuda_cflags'], spec['ldflags']]}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _directory(name):
    return paths.BUILD_DIR / name


def _import(name, library):
    """Load the library as module `name` without registering it in sys.modules (the wrapper packages own those names)."""
    import torch  # noqa: F401  (loads libc10 / libtorch, which the extension links against)
    loader = importlib.machinery.ExtensionFileLoader(name, str(library))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


def load(name):
    """The extension module; built first when its library is missing or was built from other sources."""
    if name not in _LOADED:
        directory, wanted = _directory(name), digest(name)
        library, stamp = directory / f'{name}.so', directory / 'SOURCES.sha256'
        if library.exists() and stamp.exists() and stamp.read_text().strip() == wanted:
            _LOADED[name] = _import(name, library)
        else:
            _LOADED[name] = build(name)
    return _LOADED[name]


def build(name):
    from torch.utils import cpp_extension
    spec = SPECS[name]()
    directory = _directory(name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'SOURCES.sha256').unlink(missing_ok=True)
    print(f'abundance_gs: building CUDA extension {name} in {directory} (first use; can take minutes)', flush=True)
    saved = sys.modules.get(name)
    module = cpp_extension.load(name=name, sources=spec['sources'], extra_include_paths=spec['include'],
                                extra_cflags=spec['cflags'], extra_cuda_cflags=spec['cuda_cflags'],
                                extra_ldflags=spec['ldflags'], build_directory=str(directory), is_python_module=True,
                                verbose=False)
    if saved is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved
    (directory / 'SOURCES.sha256').write_text(digest(name) + '\n')
    return module


def adopt(name, old_build, old_roots):
    """Reuse a library built elsewhere: every source and header must be byte-identical to ours.

    old_build: the library, or a directory holding <name>.so; old_roots: directories searched, in order, for each
    relative source path.
    """
    spec = SPECS[name]()
    mismatched = []
    for relative, ours in _files(spec).items():
        theirs = next((Path(r) / relative for r in old_roots if (Path(r) / relative).exists()), None)
        if theirs is None or _sha(theirs) != _sha(ours):
            mismatched.append(relative)
    if mismatched:
        raise ValueError(f'{name}: {len(mismatched)} files differ from the old build, e.g. {mismatched[:5]}')
    directory = _directory(name)
    directory.mkdir(parents=True, exist_ok=True)
    library = Path(old_build) if Path(old_build).is_file() else Path(old_build) / f'{name}.so'
    shutil.copy2(library, directory / f'{name}.so')
    (directory / 'SOURCES.sha256').write_text(digest(name) + '\n')
    (directory / 'ADOPTED_FROM').write_text(f'{old_build}\n')
    return directory


def run(args):
    """Stage `build`: build (or with --adopt-from, adopt) the CUDA extensions."""
    names = args.extensions or list(SPECS)
    for name in names:
        if args.adopt_from:
            print(name, 'adopted into', adopt(name, args.adopt_from, args.old_root))
        else:
            load(name)
            print(name, 'ready in', _directory(name))

"""tiny-cuda-nn runtime setup (environment only, no renderer change)."""
from pathlib import Path


def configure_tcnn():
    import tinycudann
    import tinycudann.modules as modules
    # The source wheel omits the RTC headers; use those of the checkout the extension was compiled from.
    path = Path(modules.__file__).parent / 'rtc/include'
    assert (path / 'tiny-cuda-nn/ministd.h').is_file()
    assert (path / 'cuda_fp16.h').is_file()
    modules._C.rtc_set_include_dir(str(path))
    # Driver 550 cannot load PTX emitted by the wheel's CUDA 12.6 NVRTC; the precompiled sm_86 FullyFusedMLP kernels
    # are used, and the optional RTC fusion is disabled explicitly rather than silently falling back.
    tinycudann.supports_jit_fusion = lambda *args, **kwargs: False

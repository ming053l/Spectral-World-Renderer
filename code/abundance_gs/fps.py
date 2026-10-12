"""Stage `fps`: render speed of a trained decoder over the TEST views of a pack, alone on its GPU.

Per view: hit tracing (two rasteriser passes), per-hit decode and compositing, i.e. spectra and abundances.
`reference`: FP32, every hit (the in-training number). `fast`: MLP in fp16, hits of blend weight < 1e-4 skipped,
endmember mixing after compositing (the paper's FPS; same scores). `fused` (opt-in): the fast renderer with the
fused CUDA kernels of csrc/fused_render.cu. Three passes, the first discarded.
"""
import json

import torch

from .inference import ViewRenderer, open_model, render_times


def run(args):
    reference, model, splats, pack, hits = open_model(args.model, args.pack)
    renderers = {'reference': reference, 'fast': ViewRenderer(model, hits, pack, fast=True, tau=args.tau),
                 'fused': ViewRenderer(model, hits, pack, fast='fused', tau=args.tau)}
    modes = {'both': ['reference', 'fast'], 'all': ['reference', 'fast', 'fused']}.get(args.mode, [args.mode])
    out = {'resolution_wh': [pack.width, pack.height], 'bands': pack.bands, 'materials': model.num_materials,
           'primitives': len(splats['means']), 'gpu': torch.cuda.get_device_name(), 'test_views': len(pack.test),
           'what': 'trace (2 rasteriser passes) + per-hit decode + compositing; spectra and abundances'}
    for mode in modes:
        torch.cuda.reset_peak_memory_stats()
        times = render_times(renderers[mode], pack.test)
        out[mode] = {'fps': len(times) / sum(times), 'ms_per_view': 1000 * sum(times) / len(times), 'views_timed': len(times),
                     'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                     'renderer': {'reference': 'fp32, all hits', 'fast': f'fp16 MLP, skip weight < {args.tau:g}'}.get(
                         mode, f'fp16 MLP, skip weight < {args.tau:g}, fused kernels')}
    out['hits_per_pixel'] = sum(hits.hits_per_pixel.values()) / len(hits.hits_per_pixel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))

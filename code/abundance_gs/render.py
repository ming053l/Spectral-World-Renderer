"""Stage `render`: write spectra and abundance maps of a trained decoder for chosen views of a pack.

<output>/<view>.npy            spectral cube [H, W, B] float32 (x hsi_normalization_scale with --radiance)
<output>/<view>_abundance.npz  abundance [K, H, W] float16, alpha [H, W] float16
<output>/endmembers.npy        K x B endmembers (x scale with --radiance)
"""
import numpy as np
import torch

from .inference import BackgroundWriter, abundance_npz, cube, open_model


def run(args):
    renderer, model, _, pack, _ = open_model(args.model, args.pack, 'fused' if args.fused else args.fast)
    if args.views == 'figures' and not args.figure_views:
        raise ValueError("--views figures needs a --scene with figure views")
    views = pack.select(','.join(args.figure_views) if args.views == 'figures' else args.views)
    scale = float(pack.meta['hsi_normalization_scale']) if args.radiance else 1.
    args.output.mkdir(parents=True, exist_ok=True)
    s = args.stride
    with BackgroundWriter() as writer:                 # compression overlaps the next render
        for i in views:
            name = pack.names[i]
            hsi, abundance, alpha = renderer(i)
            if not args.no_spectra:
                spectra = (cube(hsi, pack).clamp_min(0) * scale).permute(1, 2, 0)[::s, ::s]
                writer.submit(np.save, args.output / f'{name}.npy', spectra.float().cpu().numpy())
            writer.submit(np.savez_compressed, args.output / f'{name}_abundance.npz', **abundance_npz(abundance, alpha, pack, stride=s))
            print(name, flush=True)
    with torch.no_grad():
        np.save(args.output / 'endmembers.npy', model.endmembers().float().cpu().numpy() * scale)

"""Stage `viewer`: an interactive web viewer of a trained model (viser, in the manner of the nerfstudio viewer).

    abgs viewer --scene market [--ckpt model_ema.pt | --ckpt market/std_K24 --registry models.json] [--port 7007]

Modules: session.py (model, cameras, rendering, display modes, readout, hit inspection, exports; no GUI),
queries.py (per-Gaussian attributes and material queries), jobs.py (train / eval subprocesses), app.py (the GUI).
The server listens on --host (default 127.0.0.1); from another machine: ssh -L 7007:localhost:7007 <host>.
"""
import json
import time
from pathlib import Path


def resolve_model(model, registries):
    """A model path, or a registry entry (name -> model_ema.pt in one of the --registry files)."""
    if model is None or Path(model).exists():
        return model
    for path in registries:
        registry = json.loads(Path(path).read_text())
        if model in registry:
            return registry[model]
    raise SystemExit(f'--ckpt {model}: neither a file nor an entry of ' + (', '.join(map(str, registries)) or 'a --registry'))


def run(args):
    from .app import ViewerApp
    from .session import Session
    session = Session(args.workdir, renderer=args.renderer)
    session.extra_libraries = list(args.library)
    explicit = 'model' not in args.scene_filled          # --ckpt given: the scene's evaluation is not its evaluation
    session.load(scene=args.scene, model=resolve_model(args.model, args.registry), pack=args.pack, train_pack=args.train_pack,
                 eval_dir=None if explicit and 'eval_dir' in args.scene_filled else args.eval_dir, dataset=args.dataset,
                 label_space=args.label_space)
    app = ViewerApp(session, host=args.host, port=args.port, registries=args.registry)
    print(f'SWR viewer on http://{args.host}:{args.port}  (remote: ssh -L {args.port}:localhost:{args.port} <host>)', flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        for job in app.jobs:
            job.stop()
        app.server.stop()

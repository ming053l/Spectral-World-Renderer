"""The viewer's state without a GUI: a loaded model, free or dataset cameras, rendering, display modes, per-pixel
readout, hit inspection, material queries and exports. `app.py` binds it to the viser GUI; tests drive it directly.

Rendering reuses the package renderers: a free camera is wrapped as a one-view pack (`CameraPack`), its hit lists are
traced once by `model.Hits` and kept with the frame, and `render_view_fused` / `render_view_fast` / `render_view`
composite them. The frame keeps the composited mixing coefficients sum_i w_i s_i a_i [P, K], so spectra with some
endmembers switched off are recomputed without rendering again (spectrum = coefficients @ E).
"""
import io
import json
import math
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import records, scenes
from ..data import Pack
from ..defaults import Render
from ..model import Hits, directional_code, load_model, render_view, render_view_fast, render_view_fused, select_pixels
from . import queries as Q

# scene configuration keys read by the viewer (stage option -> dotted key), also used by --scene in cli.py
SCENE_KEYS = {'model': 'paper_model.model', 'pack': 'packs.eval', 'train_pack': 'packs.train',
              'eval_dir': 'paper_model.eval_dir', 'dataset': 'data.noisy_dataset', 'label_space': 'data.label_space'}

MODES = ['pseudo-RGB', 'band', 'false colour', 'abundance', 'dominant endmember', 'class map', 'depth', 'opacity',
         'dataset RGB']
FALSE_COLOUR = {'colour infrared (860 / 650 / 550 nm)': (860, 650, 550), 'red edge (740 / 705 / 670 nm)': (740, 705, 670),
                'NIR / green / blue (900 / 550 / 460 nm)': (900, 550, 460), 'visible (640 / 550 / 460 nm)': (640, 550, 460)}


# ---------------------------------------------------------------- cameras

@dataclass
class Camera:
    """Pinhole camera, c2w in OpenCV axes (+x right, +y down, +z forward) of the scene's world frame."""
    c2w: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    name: str = 'free'

    @classmethod
    def from_pack(cls, pack, i):
        v = pack.views[i]
        return cls(np.asarray(v['c2w'], dtype=np.float64), v['fx'], v['fy'], v['cx'], v['cy'], v['width'], v['height'],
                   pack.names[i])

    @classmethod
    def from_pose(cls, R, t, fov, aspect, max_side):
        """A camera of vertical field of view `fov` (radians) and aspect width/height, longest side max_side."""
        width, height = (max_side, max(8, round(max_side / aspect))) if aspect >= 1 else (max(8, round(max_side * aspect)), max_side)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, t
        fy = height / 2 / math.tan(fov / 2)
        return cls(c2w, fy, fy, width / 2, height / 2, int(width), int(height))

    @property
    def fov(self):
        return 2 * math.atan(self.height / 2 / self.fy)

    @property
    def aspect(self):
        return self.width / self.height

    def resized(self, max_side):
        s = max_side / max(self.width, self.height)
        if abs(s - 1) < 1e-6:
            return self
        w, h = max(8, round(self.width * s)), max(8, round(self.height * s))
        return Camera(self.c2w, self.fx * w / self.width, self.fy * h / self.height, self.cx * w / self.width,
                      self.cy * h / self.height, w, h, self.name)


class CameraPack:
    """One camera with the part of the `Pack` interface that Hits and the renderers use (view index 0)."""

    def __init__(self, camera):
        self.camera, self.width, self.height = camera, camera.width, camera.height
        self.names = [camera.name]

    def K(self, i):
        c = self.camera
        return torch.tensor([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], dtype=torch.float32)

    def c2w(self, i):
        return torch.tensor(self.camera.c2w, dtype=torch.float32)

    def center(self, i, device='cuda'):
        return self.c2w(i)[:3, 3].to(device)


def interpolate_path(keyframes, frames):
    """Cameras along a path through keyframes: positions on a Catmull-Rom-like cubic spline (linear for two),
    orientations by spherical interpolation, field of view linearly."""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp
    n = len(keyframes)
    if n < 2:
        raise ValueError('a camera path needs at least two keyframes')
    t = np.arange(n, dtype=np.float64)
    s = np.linspace(0, n - 1, frames)
    pos = np.stack([k.c2w[:3, 3] for k in keyframes])
    position = CubicSpline(t, pos, bc_type='natural')(s) if n > 2 else (pos[0] + (pos[1] - pos[0]) * s[:, None])
    rotation = Slerp(t, Rotation.from_matrix(np.stack([k.c2w[:3, :3] for k in keyframes])))(s).as_matrix()
    fov = np.interp(s, t, [k.fov for k in keyframes])
    ref = keyframes[0]
    return [Camera.from_pose(rotation[j], position[j], fov[j], ref.aspect, max(ref.width, ref.height)) for j in range(frames)]


# ---------------------------------------------------------------- the world of a dataset and the class map

class World:
    """What the dataset knows beyond the views: band centres, sensor (pseudo-RGB), the spectral library and, for
    SWR-Bench, the direct and sky irradiances Phi_d, Phi_s (least squares over the specimens: v^u_j ~ rho_j Phi_u)."""

    def __init__(self, dataset, bands):
        self.dataset = Path(dataset) if dataset else None
        self.bands = bands
        self.illumination = False
        self.library = []                      # (name, spectrum [B], class name or '')
        self.wavelengths = np.linspace(420., 980., bands)
        self.nominal_wavelengths = True
        self.rgb = None                        # 3 x B projection to linear sRGB
        bank = self.dataset / 'gt/spectral_sample_bank.npz' if self.dataset else None
        if bank is not None and bank.exists():
            b = np.load(bank)
            if len(b['band_centers_nm']) == bands:
                self.wavelengths, self.nominal_wavelengths = b['band_centers_nm'].astype(np.float64), False
                rho = b['endmembers_reflectance'].astype(np.float64)
                vd, vs = b['endmembers_direct'].astype(np.float64), b['endmembers_sky'].astype(np.float64)
                self.phid = (vd * rho).sum(1) / (rho * rho).sum(1)
                self.phis = (vs * rho).sum(1) / (rho * rho).sum(1)
                self.whitener = Q.Whitener(self.phid, self.phis)
                self.illumination = True
                classes = self._classes(b['material_id'])
                for j in range(rho.shape[1]):
                    family, species = str(b['sample_family'][j]), str(b['sample_species'][j])
                    name = f"specimen {j:02d} {family}{' ' + species if species else ''}"
                    self.library.append((name + (f' [{classes[j]}]' if classes[j] else ''), rho[:, j], classes[j]))
        sensor = self.dataset / 'gt/sensor.npz' if self.dataset else None
        if sensor is not None and sensor.exists() and not self.nominal_wavelengths:
            s = np.load(sensor)
            cmf = np.stack([np.interp(self.wavelengths, s['wavelength_nm'], s['cie_xyz'][:, k]) for k in range(3)], 1)
            srgb = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                             [0.0556434, -0.2040259, 1.0572252]])
            self.rgb = srgb @ s['xyz_adaptation'] @ cmf.T
        if self.rgb is None:                   # Gaussian windows around 610 / 550 / 465 nm
            self.rgb = np.stack([np.exp(-.5 * ((self.wavelengths - c) / 30) ** 2) for c in (610, 550, 465)])
        self.rgb = self.rgb / self.rgb.sum(1, keepdims=True)

    def _classes(self, material_id):
        protocol = self.dataset / 'material_class_protocol.json'
        if not protocol.exists():
            return [''] * len(material_id)
        p = json.loads(protocol.read_text())
        names, m2c = p.get('primary_classes'), p.get('material_id_to_primary_class')
        if not names:
            return [''] * len(material_id)
        return [names[m2c[m] if m2c else m] for m in material_id]

    def add_library(self, path):
        """Extra library spectra: .npz with rho [B, M] and name [M] (e.g. fingerprint_query_v1 ECOSTRESS caches), or
        a CSV of one spectrum (parse_csv)."""
        path = Path(path)
        if path.suffix == '.npz':
            z = np.load(path)
            rho = z['rho']
            names = [str(n) for n in z['name']] if 'name' in z.files else [f'{path.stem} {m}' for m in range(rho.shape[1])]
            for m in range(rho.shape[1]):
                self.library.append((f'{path.stem}: {names[m]}', rho[:, m].astype(np.float64), ''))
        else:
            self.library.append((path.name, self.parse_csv(path.read_bytes()), ''))

    def parse_csv(self, content):
        """A spectrum from CSV text: two columns (wavelength in nm or um, value; a header line is skipped) resampled to
        the band centres, or one column / row of B values. Percent values (> 1.5) are divided by 100."""
        rows = []
        for line in content.decode('utf-8', 'replace').replace(';', ',').splitlines():
            parts = [p for p in line.replace('\t', ',').replace(' ', ',').split(',') if p]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
        if not rows:
            raise ValueError('no numeric rows in the CSV')
        if all(len(r) >= 2 for r in rows) and len(rows) > 2:
            a = np.array([r[:2] for r in rows])
            x, y = a[np.argsort(a[:, 0]), 0], a[np.argsort(a[:, 0]), 1]
            if x.max() < 50:
                x = x * 1000.
            y = np.interp(self.wavelengths, x, y)
        else:
            y = np.array([v for r in rows for v in r], dtype=np.float64)
            if len(y) != self.bands:
                raise ValueError(f'{len(y)} values, expected {self.bands} bands or (wavelength, value) rows')
        if y.max() > 1.5:
            y = y / 100.
        return np.clip(y, 0, None)

    def band(self, wavelength):
        return int(np.abs(self.wavelengths - wavelength).argmin())


class Mapping:
    """Endmember -> class map of a model (the material scorer's TRAIN-view calibration, eval
    benchmark_material_evaluation.json) and the endmember angles to the nearest reference spectrum."""

    def __init__(self, eval_dir, K, label_space=None):
        self.channel_to_class, self.class_names, self.angles, self.space, self.source = None, [], None, label_space, None
        path = Path(eval_dir) / 'benchmark_material_evaluation.json' if eval_dir else None
        if path is None or not path.exists():
            return
        e = json.loads(path.read_text())
        spaces = e['calibration']['label_spaces']
        space = label_space if label_space in spaces else next(iter(spaces))
        c2c = spaces[space]['channel_to_class']
        if len(c2c) != K:
            return
        self.space, self.source = space, path
        self.channel_to_class = np.array([-1 if c is None else c for c in c2c])
        self.class_names = e['evaluation']['label_spaces'][space]['mapped_dominant_slot_material_agreement']['class_names']
        try:
            per = e['endmember_diagnostics']['label_spaces'][space]['per_channel']
            self.angles = np.array([p.get('nearest_direct_or_sky_SAM_degrees', np.nan) for p in per], dtype=np.float64)
        except KeyError:
            pass

    @property
    def available(self):
        return self.channel_to_class is not None

    def name(self, k):
        if not self.available or self.channel_to_class[k] < 0:
            return '-'
        return self.class_names[self.channel_to_class[k]]

    def channels(self, class_name):
        c = self.class_names.index(class_name)
        return [int(k) for k in np.where(self.channel_to_class == c)[0]]


# ---------------------------------------------------------------- colours and plots

def palette(n, name='tab20'):
    import matplotlib
    base = matplotlib.colormaps[name].colors if n <= len(matplotlib.colormaps[name].colors) else \
        list(matplotlib.colormaps['tab20'].colors) + list(matplotlib.colormaps['tab20b'].colors) + list(matplotlib.colormaps['tab20c'].colors)
    return (np.array([base[k % len(base)] for k in range(n)])[:, :3] * 255).astype(np.uint8)


def colormap(x, name='viridis'):
    import matplotlib
    return (matplotlib.colormaps[name](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def srgb(linear):
    x = np.clip(linear, 0, 1)
    return (np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - .055) * 255 + .5).astype(np.uint8)


def figure_image(fig, dpi=100):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))


def save_png(path, image):
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return Path(path)


# ---------------------------------------------------------------- the session

@dataclass
class Frame:
    """One render: camera, its hit lists and the composited maps (flat, on the GPU)."""
    camera: Camera
    hits: dict
    hsi: torch.Tensor            # [P, B]
    abundance: torch.Tensor      # [P, K]
    alpha: torch.Tensor          # [P]
    coefficients: torch.Tensor   # [P, K] sum_i w_i s_i a_i
    seconds: float
    view: int = -1               # dataset view index (-1: free camera)


class Session:
    """A loaded model with its scene; every GPU call is serialized by `lock`."""

    def __init__(self, workdir, renderer='fused', device='cuda'):
        self.workdir = Path(workdir).expanduser()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.renderer, self.device = renderer, device
        self.lock = threading.RLock()
        self.model = None
        self.queries, self.active = {}, None
        self.attributes = None
        self.exposure = None
        self.config = {}
        self.extra_libraries = []

    # ------------------------------------------------------------ loading
    def load(self, scene=None, model=None, pack=None, train_pack=None, eval_dir=None, dataset=None, label_space=None):
        """Scene configuration values fill what is not given; the model may be a path or a registry name."""
        given = dict(model=model, pack=pack, train_pack=train_pack, eval_dir=eval_dir, dataset=dataset, label_space=label_space)
        config = scenes.load(scene) if scene else {}
        paper = scenes.get(config, SCENE_KEYS['model'])
        for key, dotted in SCENE_KEYS.items():
            # the scene's evaluation (class map) belongs to the paper's model only
            if given[key] is None and not (key == 'eval_dir' and model is not None and paper and Path(model) != Path(paper)):
                given[key] = scenes.get(config, dotted)
        if given['model'] is None or given['pack'] is None:
            raise ValueError('a model and a pack are needed (give --scene or --model and --pack)')
        with self.lock:
            self.scene, self.config, self.paths = scene, config, given
            self.pack = Pack(given['pack'])
            self.train_pack = Pack(given['train_pack']) if given['train_pack'] else self.pack
            self.load_model(given['model'], eval_dir=given['eval_dir'])
            self.world = World(given['dataset'], self.model.num_bands)
            for path in self.extra_libraries:
                self.world.add_library(path)
        return self

    def load_model(self, path, eval_dir=None):
        """(Re)load a decoder on the same scene, e.g. the latest checkpoint of a running training job."""
        with self.lock:
            self.model_path = Path(path)
            self.model, self.splats = load_model(path, self.device)
            self.E = self.model.endmembers().detach().float()
            K = self.model.num_materials
            self.enabled = torch.ones(K, dtype=torch.bool, device=self.device)
            if eval_dir is None and self.paths.get('model') and Path(self.paths['model']) == Path(path):
                eval_dir = self.paths.get('eval_dir')
            local_eval = self.model_path.parent / 'eval'
            if eval_dir is None and local_eval.exists():
                eval_dir = local_eval
            self.eval_dir = eval_dir
            self.mapping = Mapping(eval_dir, K, self.paths.get('label_space'))
            self.attributes, self.queries, self.active = None, {}, None
            self.frame = None

    def set_mapping(self, eval_dir):
        with self.lock:
            self.eval_dir = eval_dir
            self.mapping = Mapping(eval_dir, self.model.num_materials, self.paths.get('label_space'))
        return self.mapping.available

    @property
    def scale(self):
        return float(self.pack.meta.get('hsi_normalization_scale', 1.))

    def info(self):
        m = self.model
        lines = [f"**model** `{self.model_path}`",
                 f"Gaussians {len(self.splats['means'])}, endmembers K = {m.num_materials}, bands B = {m.num_bands} "
                 f"({self.world.wavelengths[0]:.0f}-{self.world.wavelengths[-1]:.0f} nm"
                 f"{', nominal' if self.world.nominal_wavelengths else ''})",
                 f"**pack** `{self.pack.directory}` ({len(self.pack.train)} train / {len(self.pack.test)} test views, "
                 f"{self.pack.width}x{self.pack.height})",
                 f"**class map** {('`' + str(self.mapping.source) + '` (' + self.mapping.space + ')') if self.mapping.available else 'none (run eval for this model)'}",
                 f"**illumination** {'direct + sky irradiance of the world (cone / whitening)' if self.world.illumination else 'unknown: plain spectral angles'}; "
                 f"library {len(self.world.library)} spectra"]
        return '  \n'.join(lines)

    # ------------------------------------------------------------ rendering
    def camera(self, view, max_side=None):
        c = Camera.from_pack(self.pack, view)
        return c.resized(max_side) if max_side else c

    @torch.no_grad()
    def render(self, camera, view=-1):
        with self.lock:
            source = Hits(self.splats, CameraPack(camera), 0)
            tic = time.perf_counter()
            hits = source.trace(0)
            source.cache[0] = hits                    # the renderers below read these hit lists
            center = torch.tensor(camera.c2w[:3, 3], dtype=torch.float32, device=self.device)
            if self.renderer == 'fused':
                hsi, abundance, alpha, coefficients = render_view_fused(self.model, source, 0, center, Render.fast_tau, mixing=True)
            elif self.renderer == 'fast':
                hsi, abundance, alpha, coefficients = render_view_fast(self.model, source, 0, center, Render.fast_tau, mixing=True)
            else:
                hsi, abundance, alpha = render_view(self.model, source, 0, center)
                coefficients = None
            torch.cuda.synchronize()
            if coefficients is None:                  # reference renderer: coefficients by least squares on E
                coefficients = hsi @ torch.linalg.pinv(self.E)
            self.frame = Frame(camera, hits, hsi, abundance, alpha[:, 0], coefficients, time.perf_counter() - tic, view)
            return self.frame

    def spectra(self, frame):
        """Rendered spectra [P, B] with the switched-off endmembers removed."""
        if bool(self.enabled.all()):
            return frame.hsi
        return (frame.coefficients * self.enabled) @ self.E

    def abundances(self, frame):
        return frame.abundance * self.enabled

    def depth(self, frame):
        """Expected depth of the hit Gaussians' centres along the camera axis, sum w z / sum w."""
        hits, cam = frame.hits, frame.camera
        n = cam.width * cam.height
        R = torch.tensor(cam.c2w[:3, :3], dtype=torch.float32, device=self.device)
        t = torch.tensor(cam.c2w[:3, 3], dtype=torch.float32, device=self.device)
        z = ((self.splats['means'] - t) @ R)[:, 2]
        pix = torch.repeat_interleave(torch.arange(n, device=self.device), hits['offsets'][1:] - hits['offsets'][:-1])
        w = hits['weight'].float()
        num = torch.zeros(n, device=self.device).index_add_(0, pix, w * z[hits['ids'].long()])
        return num / frame.alpha.clamp_min(1e-8), frame.alpha > .05

    def image(self, frame, mode='pseudo-RGB', wavelength=700., false_colour=(860, 650, 550), endmember=0,
              cmap='viridis', exposure=0.):
        """Display image uint8 [H, W, 3] of a frame."""
        H, W = frame.camera.height, frame.camera.width
        with self.lock, torch.no_grad():
            if mode == 'pseudo-RGB':
                rgb = (self.spectra(frame) @ torch.tensor(self.world.rgb.T, dtype=torch.float32, device=self.device)).cpu().numpy()
                if self.exposure is None:
                    self.exposure = float(np.percentile(rgb.max(1), 99.5)) or 1.
                out = srgb(rgb.reshape(H, W, 3) / self.exposure * 2 ** exposure)
            elif mode == 'band':
                b = self.world.band(wavelength)
                x = self.spectra(frame)[:, b].cpu().numpy()
                out = colormap(x.reshape(H, W) / max(np.percentile(x, 99.5), 1e-6) * 2 ** exposure, cmap)
            elif mode == 'false colour':
                h = self.spectra(frame)
                ch = [h[:, self.world.band(l)].cpu().numpy() for l in false_colour]
                out = np.stack([c / max(np.percentile(c, 99.5), 1e-6) for c in ch], -1).reshape(H, W, 3)
                out = srgb(out * 2 ** exposure)
            elif mode == 'abundance':
                out = colormap(self.abundances(frame)[:, endmember].cpu().numpy().reshape(H, W), cmap)
            elif mode in ('dominant endmember', 'class map'):
                a = self.abundances(frame)
                dom = a.argmax(1).cpu().numpy()
                if mode == 'dominant endmember':
                    out = palette(self.model.num_materials)[dom]
                else:
                    if not self.mapping.available:
                        raise ValueError('no endmember -> class map for this model (Eval tab: run or load an evaluation)')
                    cls = self.mapping.channel_to_class[dom]
                    out = np.where(cls[:, None] >= 0, palette(len(self.mapping.class_names), 'tab10')[np.maximum(cls, 0)], 0)
                out = (out * (frame.alpha.cpu().numpy()[:, None] > .05)).astype(np.uint8).reshape(H, W, 3)
            elif mode == 'depth':
                d, ok = self.depth(frame)
                d, ok = d.cpu().numpy(), ok.cpu().numpy()
                lo, hi = np.percentile(d[ok], [1, 99]) if ok.any() else (0, 1)
                out = colormap(1 - (d - lo) / max(hi - lo, 1e-6), 'turbo') * ok[:, None]
                out = out.astype(np.uint8).reshape(H, W, 3)
            elif mode == 'opacity':
                out = colormap(frame.alpha.cpu().numpy().reshape(H, W), 'gray')
            elif mode == 'dataset RGB':
                out = self.dataset_rgb(frame)
            else:
                raise ValueError(mode)
        return out

    def dataset_rgb(self, frame):
        """The RGB observation of a dataset view (snapped cameras only): the TRAIN pack's rgb.npy or the dataset's
        linear_rgb/<view>.npy, resized to the frame."""
        if frame.view < 0:
            raise ValueError('dataset RGB needs a dataset camera (snap to a view)')
        name = self.pack.names[frame.view]
        image = None
        linear = Path(self.paths['dataset']) / 'linear_rgb' / f'{name}.npy' if self.paths.get('dataset') else None
        if linear is not None and linear.exists():
            image = srgb(np.load(linear)[..., :3] / max(np.percentile(np.load(linear), 99.5), 1e-6))
        elif name in self.train_pack.names and (self.train_pack.directory / 'rgb.npy').exists():
            image = (np.clip(self.train_pack.rgb[self.train_pack.names.index(name)], 0, 1) * 255).astype(np.uint8)
        if image is None:
            raise ValueError(f'no RGB observation of {name}')
        from PIL import Image
        return np.asarray(Image.fromarray(image).resize((frame.camera.width, frame.camera.height)))

    def legend(self, mode):
        """(colour, label) pairs of the categorical modes."""
        K = self.model.num_materials
        if mode == 'dominant endmember':
            return [(palette(K)[k], f'e{k} {self.mapping.name(k)}') for k in range(K) if bool(self.enabled[k])]
        if mode == 'class map' and self.mapping.available:
            return [(palette(len(self.mapping.class_names), 'tab10')[c], n) for c, n in enumerate(self.mapping.class_names)]
        return []

    # ------------------------------------------------------------ queries
    def ensure_attributes(self, progress=None):
        """Per-Gaussian attributes over the TRAIN rays, cached in WORKDIR/attributes/<model digest>.npz."""
        with self.lock:
            if self.attributes is not None:
                return self.attributes
            views = Q.hsi_train_views(self.model_path, self.train_pack)
            key = records.file_digest(self.model_path)[:16] + f'_{len(views)}v_{self.train_pack.width}x{self.train_pack.height}'
            path = self.workdir / 'attributes' / f'{key}.npz'
            if path.exists():
                z = np.load(path)
                self.attributes = {k: torch.tensor(z[k], device=self.device) for k in ('weight', 'abundance', 'coefficients')}
            else:
                tic = time.perf_counter()
                self.attributes = Q.gaussian_attributes(self.model, self.splats, self.train_pack, views, progress=progress)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **{k: v.cpu().numpy() for k, v in self.attributes.items()}, views=np.array(views),
                         seconds=time.perf_counter() - tic)
            return self.attributes

    def _label(self):
        n = 1
        while f'q{n}' in self.queries:
            n += 1
        return f'q{n}'

    def add_query(self, kind, **o):
        """Run a query and make it active. kind: name (class), endmember (k), library (index), spectrum (rho, name),
        click (x, y [, frame]), index (expression, rule), combine (expression), instances (base, eps, ...)."""
        with self.lock, torch.no_grad():
            at = self.ensure_attributes()
            W = at['weight']
            label = o.get('label') or self._label()
            tic = time.perf_counter()
            handling = o.get('handling', 'auto')
            if kind == 'name':
                if not self.mapping.available:
                    raise ValueError('no endmember -> class map for this model')
                q = Q.name_query(label, at, self.mapping.channels(o['class_name']), o['class_name'], o.get('threshold', .5))
            elif kind == 'endmember':
                q = Q.endmember_query(label, at, int(o['k']), o.get('threshold', .5))
            elif kind in ('library', 'spectrum'):
                name, rho = (self.world.library[o['index']][:2] if kind == 'library' else (o.get('name', 'uploaded spectrum'), o['rho']))
                fp, how = Q.library_fingerprint(rho, self.world, handling)
                q = Q.spectrum_query(label, at, self.E, fp, how, name, W)
            elif kind == 'click':
                frame = o.get('frame') or self.frame
                x, y = int(o['x']), int(o['y'])
                h = self.spectra(frame)[y * frame.camera.width + x].double().cpu().numpy()
                fp, how = Q.click_fingerprint(h, self.world, handling)
                q = Q.click_query(label, at, self.E, fp, how, f'({x}, {y}) of {frame.camera.name}', W)
            elif kind == 'index':
                q = Q.index_query(label, at, self.E, o['expression'], o.get('rule', '>='), self.world.wavelengths, W)
            elif kind == 'combine':
                q = Q.combine(label, o['expression'], self.queries)
            elif kind == 'instances':
                base = self.queries[o['base']]
                spectral = self.queries.get(o.get('spectral')) if o.get('spectral') else None
                q = Q.instances(label, base, self.splats['means'], W, o.get('eps', .6), o.get('min_samples', 5),
                                o.get('min_size', 40), o.get('min_height', 0.), o.get('up', 2), spectral,
                                at['coefficients'] @ self.E)
            else:
                raise ValueError(kind)
            if o.get('threshold') is not None and kind not in ('name', 'endmember'):
                q.threshold = float(o['threshold'])
            torch.cuda.synchronize()
            q.seconds = time.perf_counter() - tic
            self.queries[label] = q
            self.active = label
            return q

    def rerun_instances(self, label, **changes):
        q = self.queries[label]
        p = {**q.params, **changes}
        return self.add_query('instances', label=label, base=p['base'], eps=p['eps'], min_samples=p['min_samples'],
                              min_size=p['min_size'], min_height=p['min_height'], spectral=p['spectral'])

    def selection_colours(self, query, colour=(255, 0, 255)):
        """Per-Gaussian highlight colour [N, 3] (instances: one colour per instance) and selection mask [N]."""
        sel = query.selected()
        col = torch.zeros(len(sel), 3, device=self.device)
        if query.kind == 'instances' and query.instances:
            pal = torch.tensor(palette(len(query.instances)), dtype=torch.float32, device=self.device)
            for j, m in enumerate(query.instances):
                col[torch.as_tensor(m, device=self.device)] = pal[j]
        else:
            col[:] = torch.tensor(colour, dtype=torch.float32, device=self.device)
        return col, sel

    @torch.no_grad()
    def overlay(self, frame, image, query=None, colour=(255, 0, 255), dim=.35):
        """The selected Gaussians splatted into the frame with its hit lists and blended over `image`: a pixel takes
        the highlight colour in proportion to the blend weight of selected Gaussians, sum_i w_i [i selected] / sum_i w_i."""
        query = query or (self.queries.get(self.active) if self.active else None)
        if query is None:
            return image
        with self.lock:
            col, sel = self.selection_colours(query, colour)
            hits, cam = frame.hits, frame.camera
            n = cam.width * cam.height
            pix = torch.repeat_interleave(torch.arange(n, device=self.device), hits['offsets'][1:] - hits['offsets'][:-1])
            ids = hits['ids'].long()
            w = hits['weight'].float() * sel[ids]
            share = torch.zeros(n, device=self.device).index_add_(0, pix, w) / frame.alpha.clamp_min(1e-8)
            c = torch.zeros(n, 3, device=self.device).index_add_(0, pix, w[:, None] * col[ids])
            c = c / torch.zeros(n, device=self.device).index_add_(0, pix, w).clamp_min(1e-8)[:, None]
            share = share.clamp(0, 1).cpu().numpy().reshape(cam.height, cam.width, 1)
            c = c.cpu().numpy().reshape(cam.height, cam.width, 3)
        base = image.astype(np.float32) * (dim + (1 - dim) * share)
        return np.clip(base * (1 - .6 * share) + c * .6 * share, 0, 255).astype(np.uint8)

    def selected_points(self, query=None, colour=(255, 0, 255), max_points=200000):
        query = query or self.queries[self.active]
        col, sel = self.selection_colours(query, colour)
        idx = torch.where(sel)[0][:max_points]
        return self.splats['means'][idx].cpu().numpy(), col[idx].cpu().numpy().astype(np.uint8)

    def histogram_image(self, query):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        v = query.value.float().cpu().numpy()
        w = self.attributes['weight'].cpu().numpy()
        fig, ax = plt.subplots(figsize=(4.2, 2.2))
        if query.log:
            ax.hist(np.clip(v, .05, 90), bins=np.geomspace(.05, 90, 80), weights=w, color='0.4')
            ax.set_xscale('log')
            ax.set_xlabel('angle (deg)')
        else:
            ax.hist(np.clip(v, query.lo, query.hi), bins=80, weights=w, color='0.4')
            ax.set_xlabel('value')
        ax.axvline(query.threshold, color='m', lw=1.5, label=f'threshold {query.threshold:.3g}')
        ax.axvline(query.default, color='k', lw=.8, ls='--', label=f'default {query.default:.3g}')
        ax.set_yticks([])
        ax.set_title(f'{query.label}: training-visibility-weighted histogram', fontsize=8)
        ax.legend(fontsize=7)
        return figure_image(fig)

    # ------------------------------------------------------------ pixels and hits
    @torch.no_grad()
    def readout(self, frame, x, y):
        """Rendered spectrum H(r), abundances A(r), opacity and the per-endmember contributions at a pixel."""
        p = y * frame.camera.width + x
        h = self.spectra(frame)[p].cpu().numpy()
        a = self.abundances(frame)[p].cpu().numpy()
        contributions = ((frame.coefficients[p] * self.enabled)[:, None] * self.E).cpu().numpy()
        return dict(x=x, y=y, spectrum=h, abundance=a, alpha=float(frame.alpha[p]), contributions=contributions)

    def readout_image(self, r):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        wl = self.world.wavelengths
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        order = np.argsort(-r['abundance'])[:6]
        pal = palette(self.model.num_materials) / 255.
        for k in order:
            if r['abundance'][k] < .02:
                continue
            ax.plot(wl, r['contributions'][k], color=pal[k], lw=1.2, label=f'e{k} {self.mapping.name(k)} a={r["abundance"][k]:.2f}')
        ax.plot(wl, r['spectrum'], color='k', lw=2, label='rendered H(r)')
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('normalized radiance')
        ax.set_title(f'pixel ({r["x"]}, {r["y"]}), opacity {r["alpha"]:.2f}', fontsize=8)
        ax.legend(fontsize=6, loc='upper left')
        return figure_image(fig)

    @torch.no_grad()
    def inspect(self, frame, x, y):
        """Gaussians along the ray of a pixel, front to back: id, depth, blend weight w_i, intensity s_i, abundances a_i
        (reference fp32 decoder, as render_hits), and their composite."""
        with self.lock:
            cam = frame.camera
            pix = torch.tensor([y * cam.width + x], device=self.device)
            sub, _ = select_pixels(frame.hits, pix)
            ids = sub['ids'].long()
            center = torch.tensor(cam.c2w[:3, 3], dtype=torch.float32, device=self.device)
            code = directional_code(F.normalize(center[None] - self.model.primitive_means[ids], dim=-1))
            out = self.model.decode(self.model.encode_local_features(ids, sub['xyz'].float()), code)
            R = torch.tensor(cam.c2w[:3, :3], dtype=torch.float32, device=self.device)
            depth = ((self.splats['means'][ids] - center) @ R)[:, 2]
            w = sub['weight'].float()
            a, s = out['abundance'], out['intensity'][:, 0]
            A = (w[:, None] * a).sum(0) / w.sum().clamp_min(1e-8)
            rows = [dict(order=j, gaussian=int(ids[j]), depth=float(depth[j]), weight=float(w[j]), intensity=float(s[j]),
                         dominant=int(a[j].argmax()), dominant_class=self.mapping.name(int(a[j].argmax())),
                         abundance=[round(float(v), 5) for v in a[j]]) for j in range(len(ids))]
            return dict(view=cam.name, x=x, y=y, camera_c2w=cam.c2w.tolist(), alpha=float(w.sum()), hits=rows,
                        composite_abundance=A.cpu().numpy().tolist(), composite_spectrum=(w[:, None] * out['spectrum']).sum(0).cpu().numpy().tolist(),
                        renderer_abundance=frame.abundance[pix[0]].cpu().numpy().tolist(),
                        note='w_i: blend weight of hit i (visibility); a_i, s_i: per-hit decoder outputs; '
                             'A(r) = sum w_i a_i / sum w_i, H(r) = sum w_i s_i E a_i')

    def hits_table(self, result, top=12):
        rows = sorted(result['hits'], key=lambda r: -r['weight'])[:top]
        lines = [f"**{len(result['hits'])} hits** at ({result['x']}, {result['y']}), sum w = {result['alpha']:.3f}; "
                 f"top {len(rows)} by weight", '',
                 '| # | Gaussian | depth | w_i | s_i | top abundances a_i |', '|---|---|---|---|---|---|']
        for r in rows:
            a = np.array(r['abundance'])
            top3 = ', '.join(f'e{k}:{a[k]:.2f}' for k in np.argsort(-a)[:3])
            lines.append(f"| {r['order']} | {r['gaussian']} | {r['depth']:.2f} | {r['weight']:.3f} | {r['intensity']:.2f} | {top3} |")
        return '\n'.join(lines)

    def hits_image(self, result, top=24):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        hits = [r for r in result['hits'] if r['weight'] > 1e-3][:top]
        fig, ax = plt.subplots(figsize=(4.6, 2.4))
        pal = palette(self.model.num_materials) / 255.
        if hits:
            a = np.array([r['abundance'] for r in hits]) * np.array([r['weight'] for r in hits])[:, None]
            bottom = np.zeros(len(hits))
            for k in range(a.shape[1]):
                ax.bar(range(len(hits)), a[:, k], bottom=bottom, color=pal[k], width=.85)
                bottom += a[:, k]
            ax.set_xticks(range(len(hits)), [f"{r['depth']:.1f}" for r in hits], fontsize=6, rotation=60)
        ax.set_xlabel('hit depth along the ray (front to back)')
        ax.set_ylabel('w_i a_ik')
        ax.set_title('per-hit mixing proportions (bar height w_i, colours a_i)', fontsize=8)
        return figure_image(fig)

    # ------------------------------------------------------------ endmembers
    def endmember_image(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        E = (self.E.cpu().numpy() * self.scale)
        pal = palette(len(E)) / 255.
        fig, ax = plt.subplots(figsize=(4.6, 3))
        for k, e in enumerate(E):
            ax.plot(self.world.wavelengths, e, color=pal[k], lw=1.4 if bool(self.enabled[k]) else .4,
                    ls='-' if bool(self.enabled[k]) else ':', label=f'e{k} {self.mapping.name(k)}')
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('unit-norm endmember x scale')
        ax.set_title(f'learned endmembers (K = {len(E)})', fontsize=8)
        ax.legend(fontsize=5, ncol=2)
        return figure_image(fig)

    def endmember_table(self):
        K = self.model.num_materials
        mass = self.attributes['abundance'].T @ self.attributes['weight'] if self.attributes is not None else None
        lines = ['| e | class | angle to nearest reference (deg) | training visibility share | on |', '|---|---|---|---|---|']
        for k in range(K):
            ang = self.mapping.angles[k] if self.mapping.angles is not None else float('nan')
            share = f'{float(mass[k] / mass.sum()):.3f}' if mass is not None else '-'
            lines.append(f"| e{k} | {self.mapping.name(k)} | {ang:.2f} | {share} | {'yes' if bool(self.enabled[k]) else 'no'} |")
        return '\n'.join(lines)

    # ------------------------------------------------------------ exports
    def export_cube(self, frame, path, radiance=False):
        """Spectral cube [H, W, B] float32 (.npy) and a .mat with the cube, wavelengths, abundances, opacity and E."""
        from scipy.io import savemat
        H, W = frame.camera.height, frame.camera.width
        s = self.scale if radiance else 1.
        cube = (self.spectra(frame).clamp_min(0) * s).reshape(H, W, -1).float().cpu().numpy()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix('.npy'), cube)
        savemat(path.with_suffix('.mat'), {'cube': cube, 'wavelengths_nm': self.world.wavelengths,
                                           'abundance': self.abundances(frame).reshape(H, W, -1).half().float().cpu().numpy(),
                                           'opacity': frame.alpha.reshape(H, W).cpu().numpy(),
                                           'endmembers': (self.E.cpu().numpy() * s), 'c2w': frame.camera.c2w,
                                           'K': np.array([[frame.camera.fx, 0, frame.camera.cx], [0, frame.camera.fy, frame.camera.cy], [0, 0, 1]])},
                do_compression=True)
        return path.with_suffix('.npy'), path.with_suffix('.mat')

    def export_ply(self, path, query=None):
        """The selected Gaussians as a 3DGS .ply (position, log scales, rotation wxyz, opacity logit, SH DC of their
        pseudo-RGB colour) with their mean abundances abundance_k and the query value as extra properties."""
        query = query or self.queries[self.active]
        sel = torch.where(query.selected())[0]
        s = self.splats
        rgb = (self.attributes['coefficients'][sel] @ self.E) @ torch.tensor(self.world.rgb.T, dtype=torch.float32, device=self.device)
        rgb = (rgb / max(float(torch.quantile(rgb.max(1).values, .99)), 1e-8)).clamp(0, 1)
        columns = {'x': s['means'][sel, 0], 'y': s['means'][sel, 1], 'z': s['means'][sel, 2],
                   **{f'f_dc_{c}': (rgb[:, c] - .5) / .28209479177387814 for c in range(3)},
                   'opacity': s['opacities'][sel].reshape(-1),
                   **{f'scale_{c}': s['scales'][sel, c] for c in range(3)},
                   **{f'rot_{c}': s['quats'][sel, c] for c in range(4)},
                   **{f'abundance_{k}': self.attributes['abundance'][sel, k] for k in range(self.model.num_materials)},
                   'query_value': query.value[sel].float(), 'gaussian_id': sel.float()}
        data = np.stack([v.float().cpu().numpy() for v in columns.values()], 1).astype('<f4')
        header = ['ply', 'format binary_little_endian 1.0', f'element vertex {len(data)}'] + \
                 [f'property float {k}' for k in columns] + ['end_header']
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(('\n'.join(header) + '\n').encode())
            f.write(data.tobytes())
        return path, len(data)

    def render_video(self, cameras, path, fps=30, image=None, progress=None):
        """Render cameras with `image(frame) -> uint8` and encode an H.264 mp4 with ffmpeg."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        W, H = cameras[0].width // 2 * 2, cameras[0].height // 2 * 2
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
               '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for j, cam in enumerate(cameras):
                frame = self.render(cam)
                proc.stdin.write(np.ascontiguousarray(image(frame)[:H, :W]).tobytes())
                if progress is not None:
                    progress(j + 1, len(cameras))
        finally:
            proc.stdin.close()
            proc.wait()
        if proc.returncode:
            raise RuntimeError(f'ffmpeg failed ({proc.returncode})')
        return path

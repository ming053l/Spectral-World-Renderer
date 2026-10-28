"""Material queries on the Gaussians (paper Sec. 4, "Querying primitives directly"; supplement, "Query by spectrum"
and "Object queries").

Every Gaussian i carries its mean abundance and mean spectrum over the TRAIN rays (`gaussian_attributes`),
    abar_i = sum_r w_i(r) a_i(r) / sum_r w_i(r),      hbar_i = sum_r w_i(r) h_i(r) / sum_r w_i(r),
computed once per model with the fast renderer's decoder path (fp16 MLP, hits of weight < tau skipped).
A query is a predicate on these attributes, kept as a signed margin m_i (selected <=> m_i <= 0) so that queries
combine: AND = max, OR = min, NOT = -m (`combine`). Predicates (`Query.kind`):
    name        mass of abar_i on the endmembers mapped to a class (TRAIN-view map of the material scorer) >= t
    endmember   abar_ik >= t
    spectrum    angle of hbar_i to a library / uploaded spectrum <= t; a reflectance rho is compared with the
                illumination cone {rho * (alpha Phi_d + beta Phi_s), alpha, beta >= 0}
    click       angle of hbar_i to the spectrum rendered at a clicked pixel <= t, after illumination whitening
    index       band-index formula of hbar_i (e.g. NDVI) >= t or <= t
    combine     Boolean expression over earlier queries (offset t on the combined margin)
    instances   the Gaussians of another query clustered in 3D (DBSCAN); instance count and per-instance colours
Angles are in degrees and thresholded on a log axis (margin log(theta / t)); their default threshold is Otsu's on
the histogram of log angles below 45 degrees, weighted by the Gaussians' training visibility. Without the world's
direct and sky irradiances (real captures) the cone and the whitening are not available and the plain spectral angle
is used. The comparison rules (`Fingerprint`) follow experiments/fingerprint_query_v1/query.py, which produced the
supplement's numbers.
"""
import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..defaults import Render
from ..model import Hits, directional_code

EPS_REL = 1e-4
ANGLE_EDGES = np.concatenate([np.arange(0, 5, .05), np.arange(5, 30, .25), np.arange(30, 90.001, 1.)])


# ---------------------------------------------------------------- per-Gaussian attributes over the TRAIN rays

@torch.no_grad()
def gaussian_attributes(model, splats, pack, views, tau=Render.fast_tau, chunk=1 << 20, progress=None):
    """Visibility W_i = sum_r w_i(r), mean abundance abar_i and mean mixing coefficients cbar_i = mean of s_i a_i
    (mean spectrum hbar_i = cbar_i E) of every Gaussian over the hits of `views` (decoder path of render_view_fast)."""
    device = model.features.device
    N, K = len(splats['means']), model.num_materials
    weight = torch.zeros(N, device=device)
    abundance = torch.zeros(N, K, device=device)
    coefficients = torch.zeros(N, K, device=device)
    source = Hits(splats, pack, 0)
    for n, i in enumerate(views):
        hits = source.trace(i)
        center = pack.center(i, device)
        keep = hits['weight'] >= tau
        ids_all, xyz_all, w_all = hits['ids'][keep], hits['xyz'][keep], hits['weight'][keep]
        del hits
        for start in range(0, len(ids_all), chunk):
            sl = slice(start, start + chunk)
            ids, w = ids_all[sl].long(), w_all[sl].float()
            code = directional_code(F.normalize(center[None] - model.primitive_means[ids], dim=-1))
            with torch.autocast('cuda', dtype=torch.float16):
                y = model.decoder_outputs(model.encode_local_features(ids, xyz_all[sl].float()), code).float()
            a, s = y[:, :-1].softmax(-1), F.softplus(y[:, -1:])
            weight.index_add_(0, ids, w)
            abundance.index_add_(0, ids, w[:, None] * a)
            coefficients.index_add_(0, ids, (w[:, None] * s) * a)
        if progress is not None:
            progress(n + 1, len(views))
    norm = weight.clamp_min(1e-8)[:, None]
    return {'weight': weight, 'abundance': abundance / norm, 'coefficients': coefficients / norm}


def hsi_train_views(model_path, pack):
    """The decoder's HSI TRAIN views (protocol.json next to the model), else every TRAIN view of the pack."""
    protocol = Path(model_path).parent / 'protocol.json'
    names = json.loads(protocol.read_text()).get('hsi_train_view_names') if protocol.exists() else None
    if names:
        index = {n: i for i, n in enumerate(pack.names)}
        views = [index[n] for n in names if n in index]
        if views:
            return views
    return list(pack.train)


# ---------------------------------------------------------------- spectral comparison (fingerprint_query_v1/query.py)

def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-20)


def degrees(cosine):
    return torch.rad2deg(torch.arccos(cosine.clamp(-1, 1)))


class Whitener:
    """Projection removing the constant and the top principal directions of the illumination family
    log(alpha Phi_d + Phi_s), alpha in [0, 100] (99.999 % of its variation lies in two directions)."""

    def __init__(self, phid, phis, n_pc=2):
        phid = torch.as_tensor(phid, dtype=torch.float64)
        phis = torch.as_tensor(phis, dtype=torch.float64)
        alpha = torch.cat([torch.zeros(1, dtype=torch.float64), torch.logspace(-2, 2, 41, dtype=torch.float64)])
        family = torch.log(alpha[:, None] * phid[None] + phis[None])
        family = family - family.mean(1, keepdim=True)
        family = family - family.mean(0, keepdim=True)
        _, s, vt = torch.linalg.svd(family, full_matrices=False)
        L = phid.numel()
        basis = torch.cat([torch.ones(1, L, dtype=torch.float64) / math.sqrt(L), vt[:n_pc]], 0)
        q, _ = torch.linalg.qr(basis.T)
        self.P = (torch.eye(L, dtype=torch.float64) - q @ q.T).float()
        self.explained = float((s[:n_pc] ** 2).sum() / (s ** 2).sum())

    @staticmethod
    def log(x):
        m = x.amax(-1, keepdim=True).clamp_min(1e-20)
        return torch.log(x.clamp_min(0) + EPS_REL * m)

    def angle(self, X, f):
        """RMS of the whitened log-ratio log X - log f, as arctan(rms) in degrees (about the SAM scale)."""
        d = (self.log(X) - self.log(f[None])) @ self.P.to(X.device).T
        return torch.rad2deg(torch.atan(d.norm(dim=-1) / math.sqrt(X.shape[-1])))


def cone_cosine(X, G):
    """cos of the angle between rows of X [n, L] and the non-negative cone of the columns of G [L, g], g <= 3
    (exact NNLS by enumerating the supports)."""
    Xn = unit(X)
    best = torch.full((X.shape[0],), -1.0, device=X.device, dtype=X.dtype)
    g = G.shape[1]
    for mask in range(1, 2 ** g):
        Q, R = torch.linalg.qr(G[:, [j for j in range(g) if mask >> j & 1]])
        XQ = Xn @ Q
        feasible = (XQ @ torch.linalg.inv(R).T >= -1e-9).all(1)
        best = torch.where(feasible, torch.maximum(best, XQ.norm(dim=1)), best)
    return best


class Fingerprint:
    """Generator spectra G [L, g] and a rule: 'raw' (angle to g_0), 'max' (smallest angle over the generators),
    'cone' (angle to their non-negative cone), 'white' (whitened log-ratio angle to g_0)."""

    def __init__(self, G, rule='raw', whitener=None):
        G = torch.as_tensor(np.asarray(G), dtype=torch.float32)
        self.G = G[:, None] if G.dim() == 1 else G
        self.rule, self.whitener = rule, whitener

    def angles(self, X):
        """X [n, L] non-negative spectra -> angles [n] in degrees; empty spectra get 90."""
        G = self.G.to(X.device)
        if self.rule == 'white':
            theta = self.whitener.angle(X, G[:, 0])
        elif self.rule == 'raw':
            theta = degrees(unit(X) @ unit(G[:, 0]))
        elif self.rule == 'max':
            theta = degrees((unit(X) @ unit(G, 0)).amax(1))
        elif self.rule == 'cone':
            theta = degrees(cone_cosine(X, unit(G, 0)))
        else:
            raise ValueError(self.rule)
        return torch.where(X.abs().sum(-1) > 0, theta, torch.full_like(theta, 90.))


def library_fingerprint(rho, world, handling='auto'):
    """A reflectance (or radiance) spectrum as a query: the illumination cone when the world's irradiances are known."""
    rho = np.asarray(rho, dtype=np.float64)
    if handling in ('auto', 'cone') and world.illumination:
        return Fingerprint(np.stack([rho * world.phid, rho * world.phis], 1), 'cone'), 'illumination cone'
    if handling == 'sun' and world.illumination:
        return Fingerprint(rho * (world.phid + world.phis), 'raw'), 'sunlit radiance'
    return Fingerprint(rho, 'raw'), 'spectral angle (no illumination model)'


def click_fingerprint(h, world, handling='auto'):
    """A rendered spectrum h as a query: illumination whitening (default) or the cone {h R, h, h / R}."""
    h = np.asarray(h, dtype=np.float64)
    if handling in ('auto', 'white') and world.illumination:
        return Fingerprint(h, 'white', world.whitener), 'illumination whitening'
    if handling == 'cone' and world.illumination:
        R = world.phis / (world.phid + world.phis)
        return Fingerprint(np.stack([h * R, h, h / R], 1), 'cone'), 'illumination cone of the click'
    return Fingerprint(h, 'raw'), 'spectral angle (no illumination model)'


# ---------------------------------------------------------------- thresholds

def otsu_angle(theta, weight, tmax=45.0):
    """Otsu threshold (deg) on the weighted histogram of log angles below tmax (unrelated Gaussians pile up near 90)."""
    hist, _ = np.histogram(np.clip(theta, 0, 89.999), bins=ANGLE_EDGES, weights=weight)
    keep = ANGLE_EDGES[1:] <= tmax
    h, edges = hist[keep].astype(np.float64), ANGLE_EDGES[:keep.sum() + 1]
    c = np.log(np.maximum((edges[:-1] + edges[1:]) / 2, 1e-3))
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1e-12)
    mt = (h * c).sum() / max(h.sum(), 1e-12)
    m1 = (mt * w0[-1] - m0 * w0) / np.maximum(w1, 1e-12)
    return float(edges[1:][np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def otsu_linear(x, weight, bins=400):
    lo, hi = np.percentile(x, [.5, 99.5]) if len(x) else (0., 1.)
    if hi <= lo:
        return float(lo)
    h, e = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi), weights=weight)
    h = h.astype(np.float64)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1e-12)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1e-12)
    return float(e[1:][np.argmax(w0 * w1 * (m0 - m1) ** 2)])


# ---------------------------------------------------------------- queries

@dataclass
class Query:
    """One query over the N Gaussians. `value` [N] is the queried quantity (angle in degrees, abundance mass, index
    value or, for 'combine', the combined margin); `rule` '<=' or '>=' against `threshold`."""
    label: str
    kind: str
    description: str
    value: torch.Tensor
    rule: str
    threshold: float
    default: float
    lo: float
    hi: float
    log: bool = False                     # margin on a log axis (angles)
    scale: float = .1                     # margin unit of linear quantities
    fingerprint: object = None            # spectral queries: compares instance spectra
    instances: list = field(default_factory=list)       # 'instances': index arrays of the clusters
    params: dict = field(default_factory=dict)          # what re-runs the query (e.g. the DBSCAN radius)
    seconds: float = 0.

    def margin(self, threshold=None):
        t = self.threshold if threshold is None else threshold
        if self.log:
            m = torch.log(self.value.clamp_min(1e-3) / max(t, 1e-3))
        else:
            m = (self.value - t) / self.scale
        return m if self.rule == '<=' else -m

    def selected(self, threshold=None):
        return self.margin(threshold) <= 0

    def summary(self, weight):
        sel = self.selected()
        n = int(sel.sum())
        vis = float((weight * sel).sum() / weight.sum().clamp_min(1e-12))
        text = (f'**{self.label}** ({self.kind}) {self.description}  \n'
                f'rule: value {self.rule} {self.threshold:.4g} (default {self.default:.4g})  \n'
                f'selected Gaussians: {n} of {len(sel)} ({100 * n / max(len(sel), 1):.2f} %; '
                f'{100 * vis:.2f} % of the training visibility); {1000 * self.seconds:.1f} ms')
        if self.kind == 'instances':
            text += f'  \ninstances: **{len(self.instances)}** (sizes ' + ', '.join(str(len(m)) for m in self.instances[:12]) + \
                    (' ...' if len(self.instances) > 12 else '') + ')'
        return text


def angle_query(label, kind, description, theta, weight, fingerprint=None):
    t = otsu_angle(theta.cpu().numpy(), weight.cpu().numpy())
    return Query(label, kind, description, theta, '<=', t, t, .05, 60., log=True, fingerprint=fingerprint)


def name_query(label, attributes, channels, class_name, threshold=.5):
    mass = attributes['abundance'][:, channels].sum(1) if len(channels) else torch.zeros_like(attributes['weight'])
    return Query(label, 'name', f"name '{class_name}': abundance mass on endmembers {list(channels)}", mass, '>=',
                 threshold, threshold, 0., 1.)


def endmember_query(label, attributes, k, threshold=.5):
    return Query(label, 'endmember', f'abundance of endmember {k}', attributes['abundance'][:, k], '>=',
                 threshold, threshold, 0., 1.)


def spectrum_query(label, attributes, E, fingerprint, how, name, weight):
    theta = fingerprint.angles(attributes['coefficients'] @ E)
    return angle_query(label, 'spectrum', f'{name} [{how}]', theta, weight, fingerprint)


def click_query(label, attributes, E, fingerprint, how, where, weight):
    theta = fingerprint.angles(attributes['coefficients'] @ E)
    return angle_query(label, 'click', f'click {where} [{how}]', theta, weight, fingerprint)


# band-index formulas: B(l) is the band nearest to l nm, B(l1, l2) the mean over l1..l2 nm; nd(a, b) = (a-b)/(a+b)
INDEX_PRESETS = {
    'NDVI  nd(B(780,900), B(640,690))': 'nd(B(780,900), B(640,690))',
    'red edge  B(750) / B(705)': 'B(750) / B(705)',
    'NDWI  nd(B(550), B(860))': 'nd(B(550), B(860))',
    'NIR/blue  B(800,900) / B(450,490)': 'B(800,900) / B(450,490)',
}


def band_index(expression, spectra, wavelengths):
    """Evaluate a band-index formula on spectra [N, L]; only numbers, + - * /, B(...) and nd(...) are allowed."""
    wl = np.asarray(wavelengths, dtype=np.float64)

    def B(lo, hi=None):
        if hi is None:
            return spectra[:, int(np.abs(wl - lo).argmin())]
        idx = np.where((wl >= min(lo, hi)) & (wl <= max(lo, hi)))[0]
        if not len(idx):
            idx = [int(np.abs(wl - (lo + hi) / 2).argmin())]
        return spectra[:, list(idx)].mean(1)

    def nd(a, b):
        return (a - b) / (a + b).clamp_min(1e-9)

    functions = {'B': B, 'nd': nd}
    operators = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                 ast.Div: lambda a, b: a / (b.clamp_min(1e-9) if torch.is_tensor(b) else b)}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
            return functions[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f'not allowed in a band index: {ast.dump(node)[:60]}')
    value = ev(ast.parse(expression, mode='eval'))
    if not torch.is_tensor(value):
        raise ValueError('the formula uses no band')
    return value


def index_query(label, attributes, E, expression, rule, wavelengths, weight):
    value = band_index(expression, attributes['coefficients'] @ E, wavelengths).float()
    v = value.cpu().numpy()
    t = otsu_linear(v, weight.cpu().numpy())
    lo, hi = (float(x) for x in np.percentile(v, [.5, 99.5]))
    return Query(label, 'index', f'band index {expression}', value, rule, t, t, lo, hi if hi > lo else lo + 1,
                 scale=max((hi - lo) / 10, 1e-6))


def combine(label, expression, queries):
    """Boolean expression over queries by label: and / or / not (also &, |, ~ and AND, OR, NOT), parentheses.
    AND = max of the margins, OR = min, NOT = -margin; each operand uses its own current threshold."""
    text = ' '.join({'AND': 'and', 'OR': 'or', 'NOT': 'not'}.get(tok, tok) for tok in
                    expression.replace('(', ' ( ').replace(')', ' ) ').split())
    tree = ast.parse(text, mode='eval')

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Name):
            if node.id not in queries:
                raise KeyError(f'unknown query {node.id!r}; defined: {", ".join(queries)}')
            return queries[node.id].margin()
        if isinstance(node, ast.BoolOp):
            values = [ev(v) for v in node.values]
            out = values[0]
            for v in values[1:]:
                out = torch.maximum(out, v) if isinstance(node.op, ast.And) else torch.minimum(out, v)
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            a, b = ev(node.left), ev(node.right)
            return torch.maximum(a, b) if isinstance(node.op, ast.BitAnd) else torch.minimum(a, b)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            return -ev(node.operand)
        raise ValueError(f'not allowed in a Boolean query: {ast.dump(node)[:60]}')
    margin = ev(tree)
    return Query(label, 'combine', expression, margin, '<=', 0., 0., -3., 3., scale=1.)


# ---------------------------------------------------------------- instances (object queries)

def dbscan(points, eps, min_samples):
    """DBSCAN labels (-1 = noise) with a k-d tree: core points have >= min_samples neighbours within eps (itself
    included); clusters are the connected components of the core points, border points join a neighbouring core."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(points)
    if n == 0:
        return np.zeros(0, int)
    tree = cKDTree(points)
    pairs = tree.query_pairs(eps, output_type='ndarray')
    degree = np.bincount(pairs.ravel(), minlength=n) + 1
    core = degree >= min_samples
    both = core[pairs[:, 0]] & core[pairs[:, 1]]
    p = pairs[both]
    graph = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    labels = np.where(core, component, -1)
    for a, b in pairs[core[pairs[:, 0]] != core[pairs[:, 1]]]:     # border points
        c, o = (a, b) if core[a] else (b, a)
        if labels[o] < 0:
            labels[o] = labels[c]
    return labels


def height_above_ground(means, visible, up=2, cell=8.0, bin_=.5):
    """Height of each Gaussian above the local ground: densest height bin of the lower half of the visible Gaussians
    in a cell (as in the City object queries)."""
    horizontal = [a for a in range(3) if a != up]
    key = np.floor(means[:, horizontal] / cell).astype(int)
    cells = {}
    for k, h in zip(map(tuple, key[visible]), means[visible, up]):
        cells.setdefault(k, []).append(h)
    ground = {}
    for k, v in cells.items():
        v = np.sort(v)[:max(1, len(v) // 2)]
        u, c = np.unique(np.floor(v / bin_), return_counts=True)
        ground[k] = (u[c.argmax()] + .5) * bin_
    return means[:, up] - np.array([ground.get(tuple(k), np.nan) for k in key])


def instances(label, base, means, weight, eps=.6, min_samples=5, min_size=40, min_height=0., up=2,
              spectral=None, spectra=None):
    """Instances of the Gaussians selected by `base`: visible ones (training visibility above its 20th percentile),
    optionally higher than min_height above the ground, clustered with DBSCAN; clusters of >= min_size Gaussians.
    With a spectral query `spectral`, only instances whose mean spectrum lies within its threshold are kept."""
    w = weight.cpu().numpy()
    mu = means.cpu().numpy()
    visible = w > np.percentile(w[w > 0], 20)
    candidate = visible & base.selected().cpu().numpy()
    if min_height > 0:
        candidate &= np.nan_to_num(height_above_ground(mu, visible, up), nan=-1e9) > min_height
    idx = np.where(candidate)[0]
    labels = dbscan(mu[idx], eps, min_samples)
    clusters = [idx[labels == c] for c in np.unique(labels[labels >= 0])]
    clusters = sorted((m for m in clusters if len(m) >= min_size), key=len, reverse=True)
    note = ''
    if spectral is not None and spectral.fingerprint is not None:
        kept = []
        for m in clusters:
            mt = torch.as_tensor(m, device=spectra.device)
            h = (spectra[mt] * weight[mt, None]).sum(0, keepdim=True) / weight[mt].sum().clamp_min(1e-12)
            if float(spectral.fingerprint.angles(h)[0]) <= spectral.threshold:
                kept.append(m)
        note = f'; mean spectrum within {spectral.threshold:.3g} deg of {spectral.label} ({len(kept)} of {len(clusters)})'
        clusters = kept
    value = torch.zeros(len(mu), device=weight.device)
    for m in clusters:
        value[torch.as_tensor(m, device=weight.device)] = 1.
    q = Query(label, 'instances', f'DBSCAN of {base.label} (eps {eps:g}, min_samples {min_samples}, min_size {min_size}'
              + (f', above ground > {min_height:g}' if min_height > 0 else '') + ')' + note,
              value, '>=', .5, .5, 0., 1.)
    q.instances = clusters
    q.params = dict(base=base.label, eps=eps, min_samples=min_samples, min_size=min_size, min_height=min_height,
                    spectral=None if spectral is None else spectral.label)
    return q

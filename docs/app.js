'use strict';
/* Spectral World Renderer project page.
   - Media are lazy: videos receive a source only when they come into view or are requested.
   - Only one clip player runs at a time; players pause off-screen and when the tab is hidden.
   - The overview video is a click-to-load youtube-nocookie embed. */

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const saveData = !!(navigator.connection && navigator.connection.saveData);
const $ = (s, r = document) => r.querySelector(s);
const autoplayOK = () => !reduceMotion.matches && !saveData;

/* ---------- Data ---------- */
/* Okabe-Ito colours (colour-blind safe), one per query; shared with the promo video (videos/promo_v1/G/assets/queries.json, H/qcolours.py). */
const QCOL = {orange: '#E69F00', sky: '#56B4E9', green: '#009E73', yellow: '#F0E442', blue: '#0072B2',
  vermillion: '#D55E00', purple: '#CC79A7', teal: '#3FD1C0', slate: '#9AA5B8'};
const QCOL_NAME = {orange: 'Orange', sky: 'Sky-blue', green: 'Green', yellow: 'Yellow', blue: 'Blue',
  vermillion: 'Vermilion', purple: 'Reddish-purple', teal: 'Teal', slate: 'Grey'};
/* Query modes, in display order. */
const QMODES = {
  rgb: {tag: '', aria: 'RGB rendering', right: ''},
  name: {tag: 'Name', aria: 'Name query', right: 'Abundance'},
  combo: {tag: 'Combination', aria: 'Name combination', right: 'Abundance'},
  click: {tag: 'Click', aria: 'Click query', right: 'Click query'},
  spectrum: {tag: 'Spectrum', aria: 'Reference-spectrum query', right: 'Spectrum query'},
  band: {tag: 'Band', aria: 'Band or spectral index', right: ''},
};
/* Chips: {id, label, mode, col, ...}. Clip: assets/query/<world>_<id>.mp4 (same camera path as <world>_rgb.mp4).
   click: track = assets/query/<world>_<id>.json (per-frame click position); spectrum: spec = reference curve (0-1).
   desc: one sentence on the rule. ready:false hides a chip until its clip is in assets/query/.
   @generated-begin query-list (videos/promo_v1/H/page_queries.py rewrites this block) */
const QUERY_WORLDS = [
  {key: 'nvidia', name: 'NVIDIA HQ', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'glass', label: 'glazing', mode: 'name', col: 'sky'},
    {id: 'white_paint', label: 'white paint', mode: 'name', col: 'yellow'},
    {id: 'living_leaves', label: 'vegetation', mode: 'name', col: 'green'}]},
  {key: 'forbidden_city', name: 'Forbidden City', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'marble', label: 'white marble', mode: 'name', col: 'sky'},
    {id: 'glazed_tile', label: 'yellow roof tile', mode: 'name', col: 'yellow'}]},
  {key: 'market', name: 'Market', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'terracotta', label: 'fired clay', mode: 'name', col: 'vermillion'},
    {id: 'wood', label: 'wood', mode: 'name', col: 'purple'},
    {id: 'or_stone_travertine', label: 'stone ∨ travertine', spoken: 'stone or travertine', mode: 'combo', col: 'yellow',
      desc: 'Yellow intensity shows the summed abundance of stone and travertine, the combination of two name queries.'},
    {id: 'click_wood', label: 'shadowed wood', mode: 'click', col: 'sky',
      desc: 'Click a shaded pixel (crosshair); the sunlit parts of the same material are retrieved too. The clicked spectrum, after illumination whitening, is compared with the spectrum of every Gaussian; Gaussians within 0.90° are shown in sky blue.'},
    {id: 'band_550nm', label: '553 nm', mode: 'band', col: 'slate', nm: 553},
    {id: 'band_850nm', label: '847 nm', mode: 'band', col: 'slate', nm: 847}]},
  {key: 'botanical', name: 'Botanical', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'living_leaves', label: 'vegetation', mode: 'name', col: 'green'},
    {id: 'wood', label: 'wood', mode: 'name', col: 'purple'},
    {id: 'nir_index', label: 'NIR index', mode: 'band', col: 'yellow', overlay: true,
      desc: 'The index (NIR 780–900 nm − red 640–690 nm) / (NIR + red) is computed on the spectrum of every Gaussian; Gaussians at or above 0.175 are shown in yellow.'}]},
  {key: 'karst', name: 'Karst', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'water', label: 'water', mode: 'name', col: 'blue'},
    {id: 'limestone', label: 'limestone', mode: 'name', col: 'yellow'},
    {id: 'click_limestone', label: 'shadowed limestone', mode: 'click', col: 'orange',
      desc: 'Click a shaded pixel (crosshair); the sunlit parts of the same material are retrieved too. The clicked spectrum, after illumination whitening, is compared with the spectrum of every Gaussian; Gaussians within 8.5° are shown in orange.'}]},
  {key: 'study', name: 'Study', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'paper', label: 'paper', mode: 'name', col: 'sky'},
    {id: 'wood', label: 'wood', mode: 'name', col: 'purple'}]},
  {key: 'city', name: 'City', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'living_leaves', label: 'vegetation', mode: 'name', col: 'yellow'},
    {id: 'terracotta', label: 'fired clay', mode: 'name', col: 'vermillion'},
    {id: 'spec_leaf', label: 'living-leaf spectrum', mode: 'spectrum', col: 'green', ref: 'ECOSTRESS living leaf',
      spec: [10, 11, 11, 11, 16, 23, 21, 17, 15, 13, 12, 15, 48, 85, 98, 100, 100, 99, 99, 99, 99, 99, 98, 97], specNm: [420, 980],
      desc: 'An ECOSTRESS living-leaf reflectance serves as the query. Gaussians whose spectra lie within 15.5° of its sun and sky illumination cone are shown in green.'},
    {id: 'band_550nm', label: '553 nm', mode: 'band', col: 'slate', nm: 553},
    {id: 'band_850nm', label: '847 nm', mode: 'band', col: 'slate', nm: 847}]},
];
/* @generated-end query-list */
const liveChips = w => w.chips.filter(c => c.ready !== false);

/* SWR-Bench worlds; class, rare-class and specimen counts from Table 2 of the paper. */
const WORLDS = [
  {key: 'forbidden_city', name: 'Forbidden City', setting: 'outdoor', classes: 9, rare: 3, spectra: 47},
  {key: 'market', name: 'Market', setting: 'outdoor', classes: 6, rare: 0, spectra: 29},
  {key: 'botanical', name: 'Botanical', setting: 'outdoor', classes: 6, rare: 0, spectra: 29},
  {key: 'nvidia', name: 'NVIDIA HQ', setting: 'outdoor', classes: 8, rare: 2, spectra: 27},
  {key: 'study', name: 'Study', setting: 'indoor', classes: 5, rare: 2, spectra: 5},
  {key: 'karst', name: 'Karst', setting: 'outdoor', classes: 5, rare: 2, spectra: 10},
  {key: 'city', name: 'City', setting: 'outdoor', classes: 11, rare: 5, spectra: 56},
  {key: 'office', name: 'Office', setting: 'indoor', classes: 8, rare: 3, spectra: 35},
];
const NAMES = Object.fromEntries(WORLDS.map(w => [w.key, w.name]));

/* Table 2: [PSNR, SAM, mIoU, EM] for UnMix-NeRF, 3DGS + VCA / NNLS, SWR. */
const TABLE2 = {
  forbidden_city: [[26.13, 7.37, 0.19, 27.4], [25.03, 5.05, 0.17, 30.7], [28.37, 4.45, 0.55, 4.4]],
  market:         [[31.87, 3.61, 0.13, 30.9], [30.02, 1.85, 0.24, 26.9], [33.63, 1.95, 0.78, 2.4]],
  botanical:      [[30.24, 5.99, 0.10, 38.8], [29.57, 3.12, 0.22, 27.1], [32.88, 2.97, 0.71, 2.4]],
  nvidia:         [[18.29, 14.98, 0.08, 43.0], [22.27, 12.31, 0.13, 24.4], [25.30, 8.25, 0.31, 3.7]],
  study:          [[33.78, 2.81, 0.20, 22.9], [32.53, 1.52, 0.30, 14.9], [33.14, 2.01, 0.52, 0.9]],
  karst:          [[26.91, 3.29, 0.38, 25.8], [27.19, 2.26, 0.35, 28.8], [28.10, 2.15, 0.73, 2.9]],
  city:           [[27.00, 4.40, 0.10, 30.4], [25.61, 4.21, 0.21, 29.1], [28.18, 3.53, 0.52, 3.1]],
  office:         [[25.33, 6.65, 0.11, 18.8], [25.01, 5.03, 0.17, 41.3], [27.54, 4.31, 0.53, 3.3]],
};
const METRICS = {
  psnr: {i: 0, label: 'PSNR (dB)', higher: true, digits: 2,
    note: 'PSNR on foreground pixels against clean truth; methods train on noisy views.'},
  sam: {i: 1, label: 'SAM (degrees)', higher: false, digits: 2,
    note: 'Mean spectral angle on foreground pixels; a lower SAM does not by itself imply better identification.'},
  miou: {i: 2, label: 'mIoU', higher: true, digits: 2,
    note: 'Dominant-endmember mIoU on test views. Each endmember is assigned to the class holding most of the training pixels it dominates, and this map is then frozen. No material label enters training; validation labels only select K.'},
  em: {i: 3, label: 'EM (degrees)', higher: false, digits: 1,
    note: 'Mean angle between each class’s mean spectrum over pure training pixels and its learned endmember under one-to-one assignment.'},
};

/* ---------- Shared helpers ---------- */
function tabKeys(e) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
  const tabs = [...e.currentTarget.querySelectorAll('[role="tab"]')];
  const i = tabs.indexOf(document.activeElement); if (i < 0) return;
  e.preventDefault();
  const n = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  tabs[n].focus(); tabs[n].click();
}
function selectTab(tabs, active) {
  for (const t of tabs) { const on = t === active; t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1; }
}

/* ---------- Clip players ---------- */
/* A player is {video, load(), visible, userPaused}. Only one plays at a time. */
const players = [];
function stopOthers(except) { for (const p of players) if (p !== except && !p.video.paused) p.video.pause(); }
document.addEventListener('visibilitychange', () => { if (document.hidden) for (const p of players) p.video.pause(); });
/* Posters load only when their player approaches the viewport. */
const posterIO = 'IntersectionObserver' in window ? new IntersectionObserver(es => {
  for (const e of es) if (e.isIntersecting) { const v = e.target; if (v.dataset.poster) v.poster = v.dataset.poster; v.dataset.near = '1'; posterIO.unobserve(v); }
}, {rootMargin: '600px 0px'}) : null;
function lazyPoster(v, url) {
  v.dataset.poster = url;
  if (!posterIO || v.dataset.near) { v.poster = url; return; }
  posterIO.observe(v);
}
const io = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
  for (const e of entries) {
    const p = players.find(x => x.video === e.target); if (!p) continue;
    p.visible = e.intersectionRatio >= 0.5;
    if (!p.visible) { if (!p.video.paused) p.video.pause(); continue; }
    if (autoplayOK() && !p.userPaused) { p.load(); p.video.play().catch(() => {}); }
  }
}, {threshold: [0, 0.5]}) : null;
function makePlayer(video, load) {
  const p = {video, load, visible: false, userPaused: false, ready: false};
  players.push(p);
  video.addEventListener('play', () => stopOthers(p));
  if (io) io.observe(video);
  return p;
}
/* Swap a source while keeping the playback position (clips of one world share a camera path). */
function setSource(video, src, keepTime, shouldPlay) {
  if (video.getAttribute('src') === src) return;
  const t = keepTime ? video.currentTime || 0 : 0;
  video.preload = 'auto';
  video.src = src;
  video.addEventListener('loadedmetadata', () => { if (t && t < video.duration) video.currentTime = t; }, {once: true});
  if (shouldPlay) video.play().catch(() => {});
}

/* ---------- Worlds carousel ---------- */
(function carousel() {
  const track = $('#worlds-track'), dots = $('#worlds-dots');
  for (const w of WORLDS) {
    const hasQuery = QUERY_WORLDS.some(q => q.key === w.key);
    const s = document.createElement('article');
    s.className = 'slide';
    s.innerHTML = `<img src="assets/worlds/${w.key}-400.webp" srcset="assets/worlds/${w.key}-400.webp 400w, assets/worlds/${w.key}.webp 800w" sizes="(max-width:600px) 82vw, (max-width:960px) 50vw, 300px" width="800" height="450" loading="lazy" decoding="async" alt="Rendered view of the ${w.name} world">
      <div class="body"><h3>${w.name}</h3><span class="meta">${w.setting} · ${w.classes} classes (${w.rare} rare) · ${w.spectra} specimens</span>
      <button type="button"${hasQuery ? '' : ' disabled'}>${hasQuery ? 'Query this world →' : 'Shown in the figures below'}</button></div>`;
    if (hasQuery) s.querySelector('button').addEventListener('click', () => {
      setQueryWorld(w.key);
      $('#query').scrollIntoView({behavior: reduceMotion.matches ? 'auto' : 'smooth'});
      const b = document.querySelector(`#query-worlds button[data-world="${w.key}"]`); if (b) b.focus({preventScroll: true});
    });
    track.append(s);
  }
  const slides = [...track.children];
  const perView = () => Math.max(1, Math.round(track.clientWidth / slides[0].getBoundingClientRect().width));
  const pages = () => Math.ceil(slides.length / perView());
  function renderDots() {
    dots.replaceChildren();
    for (let i = 0; i < pages(); i++) {
      const b = document.createElement('button'); b.type = 'button'; b.setAttribute('aria-label', `Show worlds, page ${i + 1}`);
      b.addEventListener('click', () => go(i)); dots.append(b);
    }
    syncDots();
  }
  function page() { return Math.round(track.scrollLeft / Math.max(1, track.scrollWidth - track.clientWidth) * (pages() - 1)) || 0; }
  function syncDots() { const p = page(); [...dots.children].forEach((d, i) => d.setAttribute('aria-current', String(i === p))); }
  function go(i) { const n = pages(); i = (i + n) % n; const idx = Math.min(slides.length - 1, i * perView()); track.scrollTo({left: slides[idx].offsetLeft - track.offsetLeft - 4, behavior: reduceMotion.matches ? 'auto' : 'smooth'}); }
  $('#worlds .prev').addEventListener('click', () => go(page() - 1));
  $('#worlds .next').addEventListener('click', () => go(page() + 1));
  track.addEventListener('scroll', () => requestAnimationFrame(syncDots), {passive: true});
  track.addEventListener('keydown', e => { if (e.key === 'ArrowRight') { e.preventDefault(); go(page() + 1); } if (e.key === 'ArrowLeft') { e.preventDefault(); go(page() - 1); } });
  addEventListener('resize', renderDots);
  renderDots();
})();

/* ---------- Overview video (YouTube, click to load) ---------- */
(function promoVideo() {
  const promo = $('#promo'), play = $('#promo-play');
  play.addEventListener('click', () => {
    const f = document.createElement('iframe');
    f.src = 'https://www.youtube-nocookie.com/embed/' + promo.dataset.yt + '?autoplay=1&rel=0&modestbranding=1&playsinline=1';
    f.title = 'Spectral World Renderer overview video';
    f.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture';
    f.allowFullscreen = true;
    promo.replaceChildren(f);
    for (const p of players) { p.video.pause(); p.userPaused = true; }
    f.focus();
  });
})();

/* ---------- Simulated observation tabs ---------- */
(function hsiTabs() {
  const tabs = [...document.querySelectorAll('#hsi-tabs [role="tab"]')];
  for (const t of tabs) t.addEventListener('click', () => {
    selectTab(tabs, t);
    const n = t.dataset.hsi;
    $('#hsi-img').src = `assets/fig/swr_hsi_${n}-1200.webp`;
    $('#hsi-zoom').dataset.full = `assets/fig/swr_hsi_${n}.webp`;
    $('#hsi-img').alt = `Simulated hyperspectral observations of ${t.textContent.replace(/ · /g, ', ')}: narrow bands, a colour-infrared composite, apparent reflectance and clean and noisy 900 nm crops.`;
  });
  $('#hsi-tabs').addEventListener('keydown', tabKeys);
})();

/* ---------- Material query with RGB comparison ---------- */
const qVideo = $('#query-video'), qRgb = $('#query-rgb'), qStatus = $('#query-status'), qPlayer = $('#query-player');
let qWorld = QUERY_WORLDS[0], qChip = liveChips(qWorld)[1];
const qp = makePlayer(qVideo, () => {
  if (qp.ready) return;
  qp.ready = true;
  applyQuerySource(false);
});
/* The RGB layer follows the main clip. */
qVideo.addEventListener('play', () => { if (!qRgb.hidden) qRgb.play().catch(() => {}); });
qVideo.addEventListener('pause', () => qRgb.pause());
qVideo.addEventListener('seeked', () => { if (!qRgb.hidden) { qRgbSeeking = true; qRgb.currentTime = qVideo.currentTime; } });
/* Keep the RGB layer running and aligned: restart it if it stalls, nudge the rate for small drift,
   and seek only for large offsets (one seek at a time, so a slow decoder is never re-seeked into a freeze). */
let qRgbSeeking = false;
qRgb.addEventListener('seeked', () => { qRgbSeeking = false; });
setInterval(() => {
  if (qRgb.hidden || qVideo.paused) { qRgb.playbackRate = 1; return; }
  if (qRgb.paused && qRgb.readyState >= 2) qRgb.play().catch(() => {});
  if (qRgb.readyState < 2 || qRgbSeeking || !qVideo.duration) return;
  const dur = qVideo.duration;
  let d = qRgb.currentTime - qVideo.currentTime;
  if (d > dur / 2) d -= dur; else if (d < -dur / 2) d += dur;   // loop wrap
  if (Math.abs(d) > 0.4) { qRgbSeeking = true; qRgb.currentTime = qVideo.currentTime % (qRgb.duration || dur); qRgb.playbackRate = 1; }
  else qRgb.playbackRate = Math.abs(d) < 0.03 ? 1 : Math.min(1.1, Math.max(0.9, 1 - d * 2));
}, 200);
qVideo.addEventListener('waiting', () => { qStatus.hidden = false; });
qVideo.addEventListener('playing', () => { qStatus.hidden = true; });
qVideo.addEventListener('canplay', () => { qStatus.hidden = true; });
qVideo.addEventListener('error', () => { qStatus.hidden = false; qStatus.textContent = 'Clip unavailable'; });

const qColour = c => QCOL[c.col] || QCOL.teal;
/* Mode glyphs (shared with the promo video): magnifier (name), crosshair (click), reference curve (spectrum),
   lambda (band / index), logical-or chevron (combination). */
const OR_SVG = '<svg class="or" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 2.5 6 10 10 2.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
function specPath(spec, w = 20, h = 14, x0 = 2, y0 = 3) {
  const n = spec.length - 1, mx = Math.max(...spec);
  return spec.map((v, i) => `${i ? 'L' : 'M'}${(x0 + w * i / n).toFixed(1)} ${(y0 + h - h * v / mx).toFixed(1)}`).join('');
}
function modeIcon(c) {
  const open = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">';
  switch (c.mode) {
    case 'click': return open + '<circle cx="12" cy="12" r="7"/><path d="M12 1.5v5M12 17.5v5M1.5 12h5M17.5 12h5"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/></svg>';
    case 'spectrum': return open + `<path d="M2 21h20" stroke-width="1.6" opacity=".6"/><path d="${specPath(c.spec || [1, 1])}"/></svg>`;
    case 'band': return '<svg viewBox="0 0 24 24" aria-hidden="true"><text x="12" y="18" text-anchor="middle" font-size="19" font-weight="700" fill="currentColor" font-family="Noto Sans, system-ui, sans-serif">λ</text></svg>';
    case 'combo': return open + '<path d="M5 5.5 12 19 19 5.5" stroke-width="3"/></svg>';
    case 'rgb': return open + '<circle cx="9" cy="9.5" r="5"/><circle cx="15" cy="9.5" r="5"/><circle cx="12" cy="15" r="5"/></svg>';
    default: return open + '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>';
  }
}
function labelHTML(text) { return text.split('∨').map(t => t.replace(/[&<>]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[ch]))).join(OR_SVG); }
const spoken = c => c.spoken || c.label;

/* Click queries: a per-frame track (fractions of the frame) places the crosshair on the clicked surface point. */
const qXhair = $('#query-xhair'), qSpec = $('#query-spec');
const tracks = {};
let qTrack = null;
function loadTrack(url) {
  if (!tracks[url]) tracks[url] = fetch(url).then(r => r.ok ? r.json() : null).catch(() => null);
  return tracks[url];
}
function placeXhair(t) {
  if (!qTrack) { qXhair.toggleAttribute('hidden', true); return; }
  const n = qTrack.xy.length, f = Math.min(n - 1, Math.max(0, Math.round(t * (qTrack.fps || 30))));
  const p = qTrack.xy[f];
  qXhair.toggleAttribute('hidden', !p);   // SVG elements have no .hidden property
  if (p) { qXhair.style.left = (p[0] / 10) + '%'; qXhair.style.top = (p[1] / 10) + '%'; }
}
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
  const onFrame = (now, meta) => { if (qTrack) placeXhair(meta.mediaTime); qVideo.requestVideoFrameCallback(onFrame); };
  qVideo.requestVideoFrameCallback(onFrame);
} else {
  const tick = () => { if (qTrack && !qVideo.paused) placeXhair(qVideo.currentTime); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}
for (const ev of ['seeked', 'loadeddata']) qVideo.addEventListener(ev, () => placeXhair(qVideo.currentTime));

function applyQuerySource(keepTime) {
  if (!qp.ready) return;
  const playing = !qVideo.paused || (qp.visible && autoplayOK() && !qp.userPaused);
  setSource(qVideo, `assets/query/${qWorld.key}_${qChip.id}.mp4`, keepTime, playing);
  if (qChip.mode !== 'rgb') {
    setSource(qRgb, `assets/query/${qWorld.key}_rgb.mp4`, false, false);
    const sync = () => { qRgb.currentTime = qVideo.currentTime; if (!qVideo.paused) qRgb.play().catch(() => {}); };
    if (qRgb.readyState >= 1) sync(); else qRgb.addEventListener('loadedmetadata', sync, {once: true});
  }
  if (qChip.mode === 'click') {
    const chip = qChip;
    loadTrack(`assets/query/${qWorld.key}_${chip.id}.json`).then(tr => { if (qChip === chip) { qTrack = tr; placeXhair(qVideo.currentTime); } });
  }
}
(function renderWorldThumbs() {
  const box = $('#query-worlds');
  for (const w of QUERY_WORLDS) {
    const b = document.createElement('button');
    b.type = 'button'; b.dataset.world = w.key;
    b.innerHTML = `<img src="assets/worlds/${w.key}-400.webp" width="400" height="225" loading="lazy" decoding="async" alt=""><span>${w.name}</span>`;
    b.setAttribute('aria-label', `World: ${w.name}`);
    b.addEventListener('click', () => setQueryWorld(w.key));
    box.append(b);
  }
})();
function setQueryWorld(key) {
  const prev = qChip;
  qWorld = QUERY_WORLDS.find(w => w.key === key) || QUERY_WORLDS[0];
  const chips = liveChips(qWorld);
  qChip = chips.find(c => c.id === prev.id && c.mode !== 'rgb')
    || (prev.mode === 'rgb' && chips[0])
    || (prev.mode !== 'name' && chips.find(c => c.mode === prev.mode))
    || chips.find(c => c.mode === 'name') || chips[0];
  for (const b of document.querySelectorAll('#query-worlds button')) b.setAttribute('aria-pressed', String(b.dataset.world === qWorld.key));
  const box = $('#query-chips'); box.replaceChildren();
  /* Chips grouped by mode; each group opens with its mode tag. */
  for (const mode of Object.keys(QMODES)) {
    const group = chips.filter(c => c.mode === mode);
    if (!group.length) continue;
    const g = document.createElement('span');
    g.className = 'qgroup'; g.setAttribute('role', 'group');
    if (QMODES[mode].tag) { g.setAttribute('aria-label', QMODES[mode].tag + ' queries'); g.innerHTML = `<span class="mode-tag" aria-hidden="true">${QMODES[mode].tag}</span>`; }
    else g.setAttribute('aria-label', 'Rendering');
    for (const c of group) {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.chip = c.id; b.className = 'qchip';
      b.style.setProperty('--qc', c.mode === 'rgb' ? '#FFFFFF' : qColour(c));
      b.innerHTML = `<span class="qi">${modeIcon(c)}</span><span class="qt">${labelHTML(c.label)}</span>`;
      if (c.mode !== 'rgb') b.setAttribute('aria-label', `${QMODES[c.mode].aria}: ${spoken(c)}`);
      b.addEventListener('click', () => { qp.userPaused = false; setChip(c.id); });
      g.append(b);
    }
    box.append(g);
  }
  lazyPoster(qVideo, `assets/query/${qWorld.key}_poster.webp`);
  lazyPoster(qRgb, `assets/query/${qWorld.key}_poster.webp`);
  setChip(qChip.id, false);
}
function chipDesc(c) {
  if (c.desc) return c.desc;
  if (c.mode === 'rgb') return 'Visible composite near 630 / 550 / 460 nm, rendered from the learned spectra.';
  if (c.mode === 'name') return `${QCOL_NAME[c.col] || 'Teal'} intensity shows the abundance of ${c.label} on a fixed 0–1 scale; all other surfaces are shown in grey.`;
  return `Single band near ${c.nm} nm; both band views share one fixed exposure, so brightness is comparable between wavelengths.` + (c.nm > 800 ? ' In the near-infrared, vegetation reflects strongly, whereas most building materials do not.' : '');
}
function setChip(id, keepTime = true) {
  const chips = liveChips(qWorld);
  qChip = chips.find(c => c.id === id) || chips[0];
  for (const b of document.querySelectorAll('#query-chips button')) b.setAttribute('aria-pressed', String(b.dataset.chip === qChip.id));
  const compare = qChip.mode !== 'rgb', col = qColour(qChip);
  qRgb.hidden = !compare; $('#query-divider').hidden = !compare; $('#query-split').hidden = !compare;
  $('#label-l').hidden = !compare; $('#label-r').hidden = !compare;
  $('#label-r').textContent = QMODES[qChip.mode].right || qChip.label;
  if (!compare) qRgb.pause();
  qPlayer.style.setProperty('--qc', compare ? col : '#FFFFFF');
  $('#query-icon').innerHTML = modeIcon(qChip);
  $('#query-term').innerHTML = qChip.mode === 'rgb' ? 'rgb' : labelHTML(qChip.label);
  /* Crosshair for click queries; small reference-curve card for spectrum queries. */
  qTrack = null; qXhair.toggleAttribute('hidden', true);
  qSpec.hidden = qChip.mode !== 'spectrum';
  if (qChip.mode === 'spectrum' && qChip.spec) {
    qSpec.querySelector('path.curve').setAttribute('d', specPath(qChip.spec, 96, 34, 4, 4));
    qSpec.querySelector('.ref').textContent = qChip.ref || 'reference';
    qSpec.querySelector('.ax').textContent = qChip.specNm ? `${qChip.specNm[0]}–${qChip.specNm[1]} nm` : '';
  }
  $('#query-desc').textContent = `${qWorld.name}: ${chipDesc(qChip)}`;
  qVideo.setAttribute('aria-label', `${qWorld.name}: ${qChip.mode === 'rgb' ? 'RGB rendering'
    : qChip.mode === 'band' ? qChip.label + (qChip.overlay ? '' : ' band')
    : qChip.mode === 'name' ? 'abundance of ' + qChip.label : QMODES[qChip.mode].aria + ', ' + spoken(qChip)}`);
  applyQuerySource(keepTime);
}
$('#query-split').addEventListener('input', e => qPlayer.style.setProperty('--split', e.target.value + '%'));
/* Clicking the player (outside the divider) toggles playback; also the way to start it without autoplay. */
qPlayer.addEventListener('dblclick', () => togglePlayer(qp));
qPlayer.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'k') { e.preventDefault(); togglePlayer(qp); } });
$('#query-chips').addEventListener('click', () => { qp.load(); if (qVideo.paused) qVideo.play().catch(() => {}); });
/* Visible play/pause buttons (WCAG 2.2.2). */
for (const [btn, p] of [[$('#query-pp'), qp], [$('#tetra-pp'), null]]) {
  btn.addEventListener('click', e => { e.stopPropagation(); togglePlayer(p || tp); });
}
function syncPP(btn, v) { const playing = !v.paused; btn.setAttribute('aria-label', playing ? 'Pause clip' : 'Play clip'); btn.textContent = playing ? '❚❚' : '▶'; }
for (const [id, v] of [['#query-pp', qVideo], ['#tetra-pp', $('#tetra-video')]]) for (const ev of ['play', 'pause']) v.addEventListener(ev, () => syncPP($(id), v));
function togglePlayer(p) {
  p.load();
  if (p.video.paused) { p.userPaused = false; p.video.play().catch(() => {}); } else { p.userPaused = true; p.video.pause(); }
}
setQueryWorld('nvidia');

/* ---------- Table 2 ---------- */
function renderTable(metric) {
  const m = METRICS[metric], body = $('#bench-table tbody');
  body.replaceChildren();
  const means = [0, 0, 0];
  for (const w of WORLDS) {
    const vals = TABLE2[w.key].map(r => r[m.i]);
    vals.forEach((v, j) => means[j] += v / WORLDS.length);
    const best = m.higher ? Math.max(...vals) : Math.min(...vals);
    const tr = document.createElement('tr');
    tr.innerHTML = `<th scope="row">${w.name}<span class="setting">${w.setting}</span></th>` +
      vals.map((v, j) => `<td class="${j === 2 ? 'ours ' : ''}${v === best ? 'best' : ''}">${v.toFixed(m.digits)}</td>`).join('');
    body.append(tr);
  }
  const tr = document.createElement('tr'); tr.className = 'mean';
  tr.innerHTML = '<th scope="row">Mean</th>' + means.map((v, j) => `<td class="${j === 2 ? 'ours' : ''}">${v.toFixed(m.digits)}</td>`).join('');
  body.append(tr);
  $('#bench-caption').textContent = `${m.label}, ${m.higher ? 'higher' : 'lower'} is better; the amber dot marks the best value in each world. ${m.note} Source: Table 2 of the paper.`;
}
(function metricTabs() {
  const tabs = [...document.querySelectorAll('#metric-tabs [role="tab"]')];
  for (const t of tabs) t.addEventListener('click', () => { selectTab(tabs, t); renderTable(t.dataset.metric); });
  $('#metric-tabs').addEventListener('keydown', tabKeys);
  renderTable('miou');
})();

/* ---------- Efficiency bars (Table 4) ---------- */
function bars(el, items, max) {
  for (const it of items) {
    const w = 100 * it.v / max, inside = w > 70;
    const row = document.createElement('div');
    row.className = 'hbar' + (it.ours ? ' ours-row' : '');
    row.innerHTML = `<span class="name">${it.name}</span><span class="track2"><span class="fill" style="width:${w}%"></span><span class="val${inside ? ' inside' : ''}" style="--w:${w}%">${it.label}</span></span>`;
    el.append(row);
  }
}
bars($('#train-bars'), [
  {name: 'SWR', v: 397, label: '6 min 37 s', ours: true},
  {name: 'UnMix-NeRF', v: 1457, label: '24 min 17 s'},
  {name: '3DGS + VCA / NNLS', v: 5102, label: '85 min 2 s'}], 5102);
bars($('#fps-bars'), [
  {name: 'SWR', v: 34.9, label: '34.9 views/s', ours: true},
  {name: '3DGS + VCA / NNLS', v: 3.6, label: '3.6 views/s'},
  {name: 'UnMix-NeRF', v: 0.18, label: '0.18 views/s'}], 34.9);

/* ---------- Tetrahedra clips ---------- */
const TETRA = ['market', 'forbidden_city', 'botanical', 'karst', 'study', 'city', 'nvidia'];
const tVideo = $('#tetra-video');
let tKey = TETRA[0];
const tp = makePlayer(tVideo, () => { if (tp.ready) return; tp.ready = true; setSource(tVideo, `assets/tetra/${tKey}.mp4`, false, false); });
(function tetraTabs() {
  const box = $('#tetra-worlds');
  for (const k of TETRA) {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.world = k; b.textContent = NAMES[k];
    b.addEventListener('click', () => setTetra(k));
    box.append(b);
  }
  box.addEventListener('keydown', tabKeys);
})();
function setTetra(k) {
  tKey = k;
  selectTab([...document.querySelectorAll('#tetra-worlds [role="tab"]')], document.querySelector(`#tetra-worlds [data-world="${k}"]`));
  lazyPoster(tVideo, `assets/tetra/${k}.webp`);
  if (tp.ready) setSource(tVideo, `assets/tetra/${k}.mp4`, false, tp.visible && autoplayOK() && !tp.userPaused);
}
tVideo.addEventListener('click', () => togglePlayer(tp));
setTetra(tKey);

/* ---------- Lightbox ---------- */
const box = $('#lightbox');
for (const z of document.querySelectorAll('.zoom')) z.addEventListener('click', () => {
  const img = box.querySelector('img');
  img.src = z.dataset.full; img.alt = z.querySelector('img').alt;
  if (box.showModal) box.showModal(); else window.open(z.dataset.full, '_blank', 'noopener');
});
box.querySelector('.close').addEventListener('click', () => box.close());
box.addEventListener('click', e => { if (e.target === box) box.close(); });

/* ---------- BibTeX copy ---------- */
$('#copy-bib').addEventListener('click', async () => {
  const text = $('#bibtex').textContent, btn = $('#copy-bib'), status = $('#copy-status');
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else { const a = document.createElement('textarea'); a.value = text; a.style.position = 'fixed'; a.style.opacity = '0'; document.body.append(a); a.select(); const ok = document.execCommand('copy'); a.remove(); if (!ok) throw Error(); }
    btn.textContent = 'Copied'; status.textContent = 'BibTeX copied to the clipboard.';
  } catch {
    const r = document.createRange(); r.selectNodeContents($('#bibtex')); const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    btn.textContent = 'Selected'; status.textContent = 'BibTeX selected; copy it with the keyboard.';
  }
  setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
});

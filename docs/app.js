'use strict';
/* Spectral World Renderer project page.
   - Lazy media: videos carry no src until they are needed.
   - Only one video plays at a time; videos pause when scrolled out of view or when the tab is hidden. */

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const saveData = !!(navigator.connection && navigator.connection.saveData);
const $ = (s, r = document) => r.querySelector(s);

/* ---------- Data ---------- */
const RGB_DESC = 'Visible composite near 630 / 550 / 460 nm, rendered from the learned spectra.';
const MAT_DESC = name => `Teal intensity shows the abundance of ${name} on a fixed 0–1 scale; all other surfaces are shown in grey.`;
const BAND_DESC = (nm, extra) => `Single band near ${nm} nm. Both band views share one fixed exposure, so brightness is comparable between wavelengths. ${extra}`;

const QUERY_WORLDS = [
  {key: 'nvidia', name: 'NVIDIA HQ', chips: [
    {id: 'rgb', label: 'RGB', kind: 'rgb'},
    {id: 'glass', label: 'glazing', kind: 'mat'},
    {id: 'white_paint', label: 'white paint', kind: 'mat'},
    {id: 'living_leaves', label: 'vegetation', kind: 'mat'}]},
  {key: 'forbidden_city', name: 'Forbidden City', chips: [
    {id: 'rgb', label: 'RGB', kind: 'rgb'},
    {id: 'marble', label: 'white marble', kind: 'mat'},
    {id: 'glazed_tile', label: 'yellow roof tile', kind: 'mat'}]},
  {key: 'market', name: 'Market', chips: [
    {id: 'rgb', label: 'RGB', kind: 'rgb'},
    {id: 'terracotta', label: 'fired clay', kind: 'mat'},
    {id: 'wood', label: 'wood', kind: 'mat'},
    {id: 'band_550nm', label: '553 nm', kind: 'band', nm: 553},
    {id: 'band_850nm', label: '847 nm', kind: 'band', nm: 847}]},
  {key: 'botanical', name: 'Botanical', chips: [
    {id: 'rgb', label: 'RGB', kind: 'rgb'},
    {id: 'living_leaves', label: 'vegetation', kind: 'mat'},
    {id: 'wood', label: 'wood', kind: 'mat'}]},
  {key: 'karst', name: 'Karst', chips: [
    {id: 'rgb', label: 'RGB', kind: 'rgb'},
    {id: 'water', label: 'water', kind: 'mat'},
    {id: 'limestone', label: 'limestone', kind: 'mat'}]},
  {key: 'study', name: 'Study', chips: [
    {id: 'rgb', label: 'RGB', kind: 'rgb'},
    {id: 'paper', label: 'paper', kind: 'mat'},
    {id: 'wood', label: 'wood', kind: 'mat'}]},
  {key: 'city', name: 'City', chips: [
    {id: 'rgb', label: 'RGB', kind: 'rgb'},
    {id: 'living_leaves', label: 'vegetation', kind: 'mat'},
    {id: 'terracotta', label: 'fired clay', kind: 'mat'},
    {id: 'band_550nm', label: '553 nm', kind: 'band', nm: 553},
    {id: 'band_850nm', label: '847 nm', kind: 'band', nm: 847}]},
];

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

/* Table 2: [PSNR, SAM, mIoU, EM] for UnMix-NeRF, 3DGS + VCA / NNLS, Abundance-GS. */
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
  psnr: {i: 0, label: 'PSNR (dB)', higher: true, digits: 2, max: 36,
    note: 'PSNR on foreground pixels against clean truth; methods train on noisy views.'},
  sam: {i: 1, label: 'SAM (degrees)', higher: false, digits: 2, max: 16,
    note: 'Mean spectral angle in degrees on foreground pixels; a lower SAM does not by itself imply better identification.'},
  miou: {i: 2, label: 'mIoU', higher: true, digits: 2, max: 1,
    note: 'Dominant-endmember mIoU on test views. Each endmember is assigned to the class holding most of the training pixels it dominates; this map is then frozen. No material label enters training; validation labels only select K.'},
  em: {i: 3, label: 'EM (degrees)', higher: false, digits: 1, max: 45,
    note: 'Mean angle between each class’s mean spectrum over pure training pixels and its learned endmember under one-to-one assignment.'},
};

/* ---------- Video manager ---------- */
const managed = new Set();
function loadSrc(video) {
  const source = video.querySelector('source[data-src]');
  if (source && !source.getAttribute('src')) { source.src = source.dataset.src; video.load(); }
}
function register(video) {
  managed.add(video);
  video.addEventListener('play', () => { for (const v of managed) if (v !== video && !v.paused) v.pause(); });
}
document.addEventListener('visibilitychange', () => { if (document.hidden) for (const v of managed) v.pause(); });

/* Clip players: muted loops that start when sufficiently visible and pause when they leave the viewport. */
const autoplayOK = () => !reduceMotion.matches && !saveData;
const visible = new Map();
const io = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
  for (const e of entries) {
    const v = e.target, show = e.intersectionRatio >= 0.5;
    visible.set(v, show);
    if (!show) { if (!v.paused) v.pause(); continue; }
    if (v.dataset.auto === '1' && autoplayOK() && v.dataset.userPaused !== '1') {
      if (!v.getAttribute('src')) v.dispatchEvent(new Event('needsrc'));
      v.play().catch(() => {});
    }
  }
}, {threshold: [0, 0.5]}) : null;

/* Swap the source of a clip player while keeping its playback position (all clips of a world share one camera path). */
function swapClip(video, src, poster, loadingEl) {
  const t = video.currentTime || 0, wasPlaying = !video.paused || (video.dataset.auto === '1' && visible.get(video) && autoplayOK());
  if (video.getAttribute('src') === src) return;
  if (poster) video.poster = poster;
  video.preload = 'auto'; /* with preload="none" a new source would not load until play() */
  video.src = src;
  if (wasPlaying) video.play().catch(() => {});
  if (loadingEl) loadingEl.hidden = false;
  video.addEventListener('loadedmetadata', () => {
    if (t && t < video.duration) video.currentTime = t;
    if (wasPlaying) video.play().catch(() => {});
  }, {once: true});
  video.addEventListener('canplay', () => { if (loadingEl) loadingEl.hidden = true; }, {once: true});
  video.addEventListener('error', () => { if (loadingEl) { loadingEl.hidden = false; loadingEl.textContent = 'Clip unavailable'; } }, {once: true});
}

/* ---------- Hero video ---------- */
const promo = $('#promo'), promoPlay = $('#promo-play');
promoPlay.addEventListener('click', () => {
  const f = document.createElement('iframe');
  f.src = 'https://www.youtube-nocookie.com/embed/' + promo.dataset.yt + '?autoplay=1&rel=0&modestbranding=1&playsinline=1';
  f.title = 'Spectral World Renderer overview video';
  f.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
  f.allowFullscreen = true;
  promo.replaceChildren(f);
  promoPlay.hidden = true;
  document.querySelectorAll('video').forEach(v => v.pause());
});

/* ---------- Material query ---------- */
const qVideo = $('#query-video'), qLoading = $('#query-loading');
qVideo.dataset.auto = '1';
register(qVideo);
let qWorld = QUERY_WORLDS[0], qChip = qWorld.chips[1];

function renderQueryTabs() {
  const tabs = $('#query-worlds');
  for (const w of QUERY_WORLDS) {
    const b = document.createElement('button');
    b.className = 'tab'; b.type = 'button'; b.setAttribute('role', 'tab');
    b.dataset.world = w.key; b.textContent = w.name;
    b.addEventListener('click', () => setQueryWorld(w.key));
    tabs.append(b);
  }
  tabs.addEventListener('keydown', tabKeys);
}
function setQueryWorld(key, chipId) {
  qWorld = QUERY_WORLDS.find(w => w.key === key) || QUERY_WORLDS[0];
  const keep = qWorld.chips.find(c => c.id === (chipId || (qChip && qChip.id)) && c.kind !== 'rgb')
    || qWorld.chips.find(c => c.kind === (qChip && qChip.kind) && c.kind === 'band') || qWorld.chips[1];
  for (const b of document.querySelectorAll('#query-worlds .tab')) {
    const on = b.dataset.world === qWorld.key;
    b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1;
  }
  const chips = $('#query-chips'); chips.replaceChildren();
  for (const c of qWorld.chips) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = `chip ${c.kind}`; b.dataset.chip = c.id;
    b.innerHTML = '<i aria-hidden="true"></i>';
    b.append(c.label);
    b.addEventListener('click', () => setChip(c.id));
    chips.append(b);
  }
  qVideo.currentTime = 0;
  qVideo.removeAttribute('src');
  setChip(keep.id, true);
}
function setChip(id, fresh) {
  qChip = qWorld.chips.find(c => c.id === id) || qWorld.chips[0];
  for (const b of document.querySelectorAll('#query-chips .chip')) b.setAttribute('aria-pressed', String(b.dataset.chip === qChip.id));
  $('#query-term').textContent = qChip.kind === 'rgb' ? 'rgb' : qChip.label;
  const legend = $('#query-legend');
  legend.hidden = qChip.kind === 'rgb';
  legend.classList.toggle('band', qChip.kind === 'band');
  let desc;
  if (qChip.kind === 'rgb') desc = RGB_DESC;
  else if (qChip.kind === 'mat') desc = MAT_DESC(qChip.label);
  else desc = BAND_DESC(qChip.nm, qChip.nm > 800 ? 'In the near-infrared, vegetation reflects strongly, whereas most building materials do not.' : 'Green light, within the visible range.');
  $('#query-desc').textContent = `${qWorld.name} · ${desc}`;
  qVideo.setAttribute('aria-label', `${qWorld.name}: ${qChip.kind === 'rgb' ? 'RGB rendering' : qChip.kind === 'band' ? qChip.label + ' band' : 'abundance of ' + qChip.label}`);
  const src = `assets/query/${qWorld.key}_${qChip.id}.mp4`, poster = `assets/query/${qWorld.key}_poster.webp`;
  if (fresh) qVideo.poster = poster;
  if (qVideo.dataset.ready === '1') swapClip(qVideo, src, poster, qLoading);
  else qVideo.dataset.pending = src;
}
qVideo.addEventListener('needsrc', () => {
  qVideo.dataset.ready = '1';
  if (qVideo.dataset.pending) { swapClip(qVideo, qVideo.dataset.pending, null, qLoading); }
});
/* Clicking the player toggles playback (also makes it usable when autoplay is disabled). */
qVideo.parentElement.addEventListener('click', () => {
  if (qVideo.dataset.ready !== '1') qVideo.dispatchEvent(new Event('needsrc'));
  if (qVideo.paused) { qVideo.dataset.userPaused = '0'; qVideo.play().catch(() => {}); }
  else { qVideo.dataset.userPaused = '1'; qVideo.pause(); }
});
renderQueryTabs();
setQueryWorld('nvidia', 'glass');
if (io) io.observe(qVideo);
/* Without autoplay, the first chip click should still start playback. */
$('#query-chips').addEventListener('click', () => {
  if (qVideo.dataset.ready !== '1') qVideo.dispatchEvent(new Event('needsrc'));
  if (!autoplayOK() && qVideo.paused) setTimeout(() => qVideo.play().catch(() => {}), 50);
});

/* ---------- Worlds gallery ---------- */
(function renderWorlds() {
  const grid = $('#worlds-grid');
  for (const w of WORLDS) {
    const card = document.createElement('article');
    card.className = 'world';
    const hasQuery = QUERY_WORLDS.some(q => q.key === w.key);
    card.innerHTML = `<img src="assets/worlds/${w.key}.webp" width="800" height="450" loading="lazy" decoding="async" alt="Rendered view of the ${w.name} world">
      <div class="body"><h3>${w.name}</h3>
      <div class="meta"><span class="io">${w.setting}</span><span>${w.classes} classes</span><span>${w.rare} rare</span><span>${w.spectra} specimens</span></div>
      <button type="button" class="go"${hasQuery ? '' : ' disabled'}>${hasQuery ? 'Query this world →' : 'Shown in the figures below'}</button></div>`;
    if (hasQuery) card.querySelector('.go').addEventListener('click', () => {
      setQueryWorld(w.key);
      $('#query').scrollIntoView({behavior: reduceMotion.matches ? 'auto' : 'smooth'});
      const tab = document.querySelector(`#query-worlds .tab[data-world="${w.key}"]`); if (tab) tab.focus({preventScroll: true});
    });
    grid.append(card);
  }
})();

/* Simulated observation figure tabs */
(function hsiTabs() {
  const tabs = [...document.querySelectorAll('#hsi-tabs [role="tab"]')];
  for (const t of tabs) t.addEventListener('click', () => {
    for (const o of tabs) { o.setAttribute('aria-selected', String(o === t)); o.tabIndex = o === t ? 0 : -1; }
    const n = t.dataset.hsi;
    $('#hsi-img').src = `assets/fig/swr_hsi_${n}-1200.webp`;
    $('#hsi-zoom').dataset.full = `assets/fig/swr_hsi_${n}.webp`;
    $('#hsi-img').alt = n === '1'
      ? 'Simulated hyperspectral observations of Forbidden City, Market, Botanical and NVIDIA HQ: narrow bands, a colour-infrared composite, apparent reflectance and clean and noisy 900 nm crops.'
      : 'Simulated hyperspectral observations of Study, Karst, City and Office: narrow bands, a colour-infrared composite, apparent reflectance and clean and noisy 900 nm crops.';
  });
  tabs.forEach((t, i) => t.tabIndex = i ? -1 : 0);
  $('#hsi-tabs').addEventListener('keydown', tabKeys);
})();

/* ---------- Table 2 ---------- */
const NAMES = Object.fromEntries(WORLDS.map(w => [w.key, w.name]));
function renderTable(metric) {
  const m = METRICS[metric], body = $('#bench-table tbody');
  body.replaceChildren();
  const fmt = v => v.toFixed(m.digits);
  const means = [0, 0, 0];
  for (const w of WORLDS) {
    const rows = TABLE2[w.key], vals = rows.map(r => r[m.i]);
    vals.forEach((v, j) => means[j] += v / WORLDS.length);
    const best = m.higher ? Math.max(...vals) : Math.min(...vals);
    const tr = document.createElement('tr');
    tr.innerHTML = `<th scope="row">${NAMES[w.key]}<span class="setting">${w.setting}</span></th>` + vals.map((v, j) => {
      const pct = Math.max(2, Math.min(100, 100 * v / m.max));
      return `<td class="${j === 2 ? 'ours ' : ''}${v === best ? 'best' : ''}"><span class="cell"><span class="bar" aria-hidden="true"><i style="width:${pct}%"></i></span>${fmt(v)}</span></td>`;
    }).join('');
    body.append(tr);
  }
  const tr = document.createElement('tr'); tr.className = 'mean';
  tr.innerHTML = `<th scope="row">Mean of eight worlds</th>` + means.map((v, j) => `<td class="${j === 2 ? 'ours' : ''}">${v.toFixed(m.digits)}</td>`).join('');
  body.append(tr);
  $('#bench-caption').textContent = `${m.label}, ${m.higher ? 'higher' : 'lower'} is better; the amber dot marks the best value in each world. ${m.note} Source: Table 2 of the paper (label-free material identification and spectral fidelity on SWR-Bench).`;
}
(function metricTabs() {
  const tabs = [...document.querySelectorAll('#metric-tabs [role="tab"]')];
  for (const t of tabs) t.addEventListener('click', () => {
    for (const o of tabs) { o.setAttribute('aria-selected', String(o === t)); o.tabIndex = o === t ? 0 : -1; }
    renderTable(t.dataset.metric);
  });
  tabs.forEach((t, i) => t.tabIndex = i ? -1 : 0);
  $('#metric-tabs').addEventListener('keydown', tabKeys);
  renderTable('miou');
})();

/* ---------- Efficiency bars (Table 4) ---------- */
function bars(el, items, max) {
  for (const it of items) {
    const w = 100 * it.v / max, inside = w > 70;
    const row = document.createElement('div');
    row.className = 'hbar' + (it.ours ? ' ours' : '');
    row.innerHTML = `<span class="name">${it.name}</span><span class="track"><span class="fill" style="width:${w}%"></span><span class="val${inside ? ' inside' : ''}" style="--w:${w}%">${it.label}</span></span>`;
    el.append(row);
  }
}
bars($('#train-bars'), [
  {name: 'Abundance-GS', v: 397, label: '6 min 37 s', ours: true},
  {name: 'UnMix-NeRF', v: 1457, label: '24 min 17 s'},
  {name: '3DGS + VCA / NNLS', v: 5102, label: '85 min 2 s'}], 5102);
bars($('#fps-bars'), [
  {name: 'Abundance-GS', v: 34.9, label: '34.9 views/s', ours: true},
  {name: '3DGS + VCA / NNLS', v: 3.6, label: '3.6 views/s'},
  {name: 'UnMix-NeRF', v: 0.18, label: '0.18 views/s'}], 34.9);

/* ---------- Tetrahedra clips ---------- */
const TETRA = ['market', 'forbidden_city', 'botanical', 'karst', 'study', 'city', 'nvidia'];
const tVideo = $('#tetra-video');
tVideo.dataset.auto = '1';
register(tVideo);
let tKey = TETRA[0];
(function tetraTabs() {
  const tabs = $('#tetra-worlds');
  for (const k of TETRA) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'tab'; b.setAttribute('role', 'tab'); b.dataset.world = k; b.textContent = NAMES[k];
    b.addEventListener('click', () => setTetra(k));
    tabs.append(b);
  }
  tabs.addEventListener('keydown', tabKeys);
})();
function setTetra(k) {
  tKey = k;
  for (const b of document.querySelectorAll('#tetra-worlds .tab')) { const on = b.dataset.world === k; b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1; }
  tVideo.poster = `assets/tetra/${k}.webp`;
  if (tVideo.dataset.ready === '1') { tVideo.preload = 'auto'; tVideo.src = `assets/tetra/${k}.mp4`; if (visible.get(tVideo) && autoplayOK()) tVideo.play().catch(() => {}); }
}
tVideo.addEventListener('needsrc', () => { tVideo.dataset.ready = '1'; tVideo.preload = 'auto'; tVideo.src = `assets/tetra/${tKey}.mp4`; });
tVideo.parentElement.addEventListener('click', () => {
  if (tVideo.dataset.ready !== '1') tVideo.dispatchEvent(new Event('needsrc'));
  if (tVideo.paused) { tVideo.dataset.userPaused = '0'; tVideo.play().catch(() => {}); } else { tVideo.dataset.userPaused = '1'; tVideo.pause(); }
});
setTetra(tKey);
if (io) io.observe(tVideo);

/* ---------- Tabs: arrow-key navigation ---------- */
function tabKeys(e) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
  const tabs = [...e.currentTarget.querySelectorAll('[role="tab"]')];
  const i = tabs.indexOf(document.activeElement); if (i < 0) return;
  e.preventDefault();
  const n = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  tabs[n].focus(); tabs[n].click();
}

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

/* ---------- Active section in the navigation ---------- */
if ('IntersectionObserver' in window) {
  const links = [...document.querySelectorAll('.nav-links a')];
  const obs = new IntersectionObserver(entries => {
    for (const e of entries) if (e.isIntersecting) for (const a of links) a.setAttribute('aria-current', String(a.getAttribute('href') === '#' + e.target.id));
  }, {rootMargin: '-45% 0px -50% 0px'});
  for (const a of links) { const s = document.querySelector(a.getAttribute('href')); if (s) obs.observe(s); }
}

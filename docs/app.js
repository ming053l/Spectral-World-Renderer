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
const NAMES = Object.fromEntries(WORLDS.map(w => [w.key, w.name]));

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
let qWorld = QUERY_WORLDS[0], qChip = qWorld.chips[1];
const qp = makePlayer(qVideo, () => {
  if (qp.ready) return;
  qp.ready = true;
  applyQuerySource(false);
});
/* The RGB layer follows the main clip. */
qVideo.addEventListener('play', () => { if (!qRgb.hidden) qRgb.play().catch(() => {}); });
qVideo.addEventListener('pause', () => qRgb.pause());
qVideo.addEventListener('seeked', () => { if (!qRgb.hidden) qRgb.currentTime = qVideo.currentTime; });
setInterval(() => {
  if (qRgb.hidden || qVideo.paused || qRgb.readyState < 2) return;
  const d = qRgb.currentTime - qVideo.currentTime;
  if (Math.abs(d) > 0.08 && Math.abs(d) < qVideo.duration - 0.5) qRgb.currentTime = qVideo.currentTime;
}, 250);
qVideo.addEventListener('waiting', () => { qStatus.hidden = false; });
qVideo.addEventListener('playing', () => { qStatus.hidden = true; });
qVideo.addEventListener('canplay', () => { qStatus.hidden = true; });
qVideo.addEventListener('error', () => { qStatus.hidden = false; qStatus.textContent = 'Clip unavailable'; });

function applyQuerySource(keepTime) {
  if (!qp.ready) return;
  const playing = !qVideo.paused || (qp.visible && autoplayOK() && !qp.userPaused);
  setSource(qVideo, `assets/query/${qWorld.key}_${qChip.id}.mp4`, keepTime, playing);
  if (qChip.kind !== 'rgb') {
    setSource(qRgb, `assets/query/${qWorld.key}_rgb.mp4`, false, false);
    const sync = () => { qRgb.currentTime = qVideo.currentTime; if (!qVideo.paused) qRgb.play().catch(() => {}); };
    if (qRgb.readyState >= 1) sync(); else qRgb.addEventListener('loadedmetadata', sync, {once: true});
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
  qChip = qWorld.chips.find(c => c.id === prev.id && c.kind !== 'rgb')
    || (prev.kind === 'band' && qWorld.chips.find(c => c.kind === 'band'))
    || (prev.kind === 'rgb' && qWorld.chips[0]) || qWorld.chips[1];
  for (const b of document.querySelectorAll('#query-worlds button')) b.setAttribute('aria-pressed', String(b.dataset.world === qWorld.key));
  const chips = $('#query-chips'); chips.replaceChildren();
  qWorld.chips.forEach((c, i) => {
    if (i && c.kind !== qWorld.chips[i - 1].kind && c.kind === 'band') { const s = document.createElement('span'); s.className = 'sep'; s.setAttribute('aria-hidden', 'true'); chips.append(s); }
    const b = document.createElement('button');
    b.type = 'button'; b.dataset.chip = c.id; b.textContent = c.label;
    b.addEventListener('click', () => { qp.userPaused = false; setChip(c.id); });
    chips.append(b);
  });
  lazyPoster(qVideo, `assets/query/${qWorld.key}_poster.webp`);
  lazyPoster(qRgb, `assets/query/${qWorld.key}_poster.webp`);
  setChip(qChip.id, false);
}
function setChip(id, keepTime = true) {
  qChip = qWorld.chips.find(c => c.id === id) || qWorld.chips[0];
  for (const b of document.querySelectorAll('#query-chips button')) b.setAttribute('aria-pressed', String(b.dataset.chip === qChip.id));
  const compare = qChip.kind !== 'rgb';
  qRgb.hidden = !compare; $('#query-divider').hidden = !compare; $('#query-split').hidden = !compare;
  $('#label-l').hidden = !compare; $('#label-r').hidden = !compare;
  $('#label-r').textContent = qChip.kind === 'band' ? qChip.label : 'Abundance';
  if (!compare) qRgb.pause();
  $('#query-term').textContent = qChip.kind === 'rgb' ? 'rgb' : qChip.label;
  let desc;
  if (qChip.kind === 'rgb') desc = 'Visible composite near 630 / 550 / 460 nm, rendered from the learned spectra.';
  else if (qChip.kind === 'mat') desc = `Teal intensity shows the abundance of ${qChip.label} on a fixed 0–1 scale; all other surfaces are shown in grey.`;
  else desc = `Single band near ${qChip.nm} nm; both band views share one fixed exposure, so brightness is comparable between wavelengths.` + (qChip.nm > 800 ? ' In the near-infrared, vegetation reflects strongly, whereas most building materials do not.' : '');
  $('#query-desc').textContent = `${qWorld.name}: ${desc}`;
  qVideo.setAttribute('aria-label', `${qWorld.name}: ${qChip.kind === 'rgb' ? 'RGB rendering' : qChip.kind === 'band' ? qChip.label + ' band' : 'abundance of ' + qChip.label}`);
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

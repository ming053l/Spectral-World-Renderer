'use strict';
/* Spectral World Renderer project page.
   - Two languages (EN / Traditional Chinese): static text carries .en/.zh spans; generated text uses L().
   - Media are lazy: videos receive a source only when they come into view or are requested.
   - Only one clip player runs at a time; players pause off-screen and when the tab is hidden.
   - The overview video is a click-to-load youtube-nocookie embed. */

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const saveData = !!(navigator.connection && navigator.connection.saveData);
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const autoplayOK = () => !reduceMotion.matches && !saveData;
const esc = s => String(s).replace(/[&<>"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[ch]));

/* ---------- Language ---------- */
const lang = () => document.documentElement.dataset.lang === 'zh' ? 'zh' : 'en';
const L = (en, zh) => `<span class="en">${en}</span><span class="zh">${zh}</span>`;   // both, CSS shows one
const T = (en, zh) => lang() === 'zh' ? zh : en;                                      // current only (attributes)
const relabel = [];
function setLang(l, save = true) {
  const h = document.documentElement;
  h.dataset.lang = l; h.lang = l === 'zh' ? 'zh-Hant' : 'en';
  for (const b of $$('.lang button')) b.setAttribute('aria-pressed', String(b.dataset.l === l));
  if (save) try { localStorage.setItem('swr-lang', l); } catch (e) {}
  const u = new URL(location.href);
  if (u.searchParams.has('lang')) { u.searchParams.set('lang', l); history.replaceState(null, '', u); }
  /* Traditional Chinese uses system fonts (PingFang TC, Microsoft JhengHei, Noto Sans CJK TC): a CJK webfont adds about 2 MB. */
  for (const f of relabel) f();
}
for (const b of $$('.lang button')) b.addEventListener('click', () => setLang(b.dataset.l));

/* ---------- Data ---------- */
/* Okabe-Ito colours (colour-blind safe), one per query; shared with the promo video. */
const QCOL = {orange: '#E69F00', sky: '#56B4E9', green: '#009E73', yellow: '#F0E442', blue: '#0072B2',
  vermillion: '#D55E00', purple: '#CC79A7', teal: '#3FD1C0', slate: '#9AA5B8'};
const QCOL_NAME = {orange: ['Orange', '橙色'], sky: ['Sky-blue', '天藍色'], green: ['Green', '綠色'], yellow: ['Yellow', '黃色'],
  blue: ['Blue', '藍色'], vermillion: ['Vermilion', '朱紅色'], purple: ['Reddish-purple', '紫紅色'], teal: ['Teal', '青綠色'], slate: ['Grey', '灰色']};
const QMODES = {
  rgb: {tag: ['', ''], aria: ['RGB rendering', 'RGB 渲染'], right: ['', '']},
  name: {tag: ['Name', '名稱'], aria: ['Name query', '名稱查詢'], right: ['Abundance', '豐度']},
  combo: {tag: ['Combination', '組合'], aria: ['Name combination', '名稱組合'], right: ['Abundance', '豐度']},
  click: {tag: ['Click', '點選'], aria: ['Click query', '點選查詢'], right: ['Click query', '點選查詢']},
  spectrum: {tag: ['Spectrum', '光譜'], aria: ['Reference-spectrum query', '參考光譜查詢'], right: ['Spectrum query', '光譜查詢']},
  band: {tag: ['Band', '波段'], aria: ['Band or spectral index', '波段或光譜指數'], right: ['', '']},
};
/* Chips: {id, label, mode, col, ...}. Clip: assets/query/<world>_<id>.mp4 (same camera path as <world>_rgb.mp4).
   click: track = assets/query/<world>_<id>.json (per-frame click position); spectrum: spec = reference curve (0-1).
   desc: [en, zh] sentence on the rule. ready:false hides a chip until its clip is in assets/query/. */
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
      desc: ['Yellow intensity shows the summed abundance of stone and travertine, the combination of two name queries.',
        '以黃色的濃淡表示 stone 與 travertine 的豐度總和，即兩個名稱查詢的組合。']},
    {id: 'click_wood', label: 'shadowed wood', mode: 'click', col: 'sky',
      desc: ['Click a shaded pixel (crosshair); the sunlit parts of the same material are retrieved too. The clicked spectrum, after illumination whitening, is compared with the spectrum of every Gaussian; Gaussians within 0.90° are shown in sky blue.',
        '點選陰影中的像素（十字準星），同一材質受日照的部分也會一併取回。點選的光譜經照明白化後，與每個高斯的光譜相比較；光譜角在 0.90° 以內的高斯以天藍色顯示。']},
    {id: 'band_550nm', label: '553 nm', mode: 'band', col: 'slate', nm: 553},
    {id: 'band_850nm', label: '847 nm', mode: 'band', col: 'slate', nm: 847}]},
  {key: 'botanical', name: 'Botanical', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'living_leaves', label: 'vegetation', mode: 'name', col: 'green'},
    {id: 'wood', label: 'wood', mode: 'name', col: 'purple'},
    {id: 'nir_index', label: 'NIR index', mode: 'band', col: 'yellow', overlay: true,
      desc: ['The index (NIR 780–900 nm − red 640–690 nm) / (NIR + red) is computed on the spectrum of every Gaussian; Gaussians at or above 0.175 are shown in yellow.',
        '對每個高斯的光譜計算指數（NIR 780–900 nm − 紅光 640–690 nm）/（NIR + 紅光）；指數不低於 0.175 的高斯以黃色顯示。']}]},
  {key: 'karst', name: 'Karst', chips: [
    {id: 'rgb', label: 'RGB', mode: 'rgb'},
    {id: 'water', label: 'water', mode: 'name', col: 'blue'},
    {id: 'limestone', label: 'limestone', mode: 'name', col: 'yellow'},
    {id: 'click_limestone', label: 'shadowed limestone', mode: 'click', col: 'orange',
      desc: ['Click a shaded pixel (crosshair); the sunlit parts of the same material are retrieved too. The clicked spectrum, after illumination whitening, is compared with the spectrum of every Gaussian; Gaussians within 8.5° are shown in orange.',
        '點選陰影中的像素（十字準星），同一材質受日照的部分也會一併取回。點選的光譜經照明白化後，與每個高斯的光譜相比較；光譜角在 8.5° 以內的高斯以橙色顯示。']}]},
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
      desc: ['An ECOSTRESS living-leaf reflectance serves as the query. Gaussians whose spectra lie within 15.5° of its sun and sky illumination cone are shown in green.',
        '以 ECOSTRESS 活體葉片反射率作為查詢。光譜落在其日光與天空光照明錐 15.5° 以內的高斯以綠色顯示。']},
    {id: 'band_550nm', label: '553 nm', mode: 'band', col: 'slate', nm: 553},
    {id: 'band_850nm', label: '847 nm', mode: 'band', col: 'slate', nm: 847}]},
];
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
const SETTING = {outdoor: ['outdoor', '室外'], indoor: ['indoor', '室內']};

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
  psnr: {i: 0, label: ['PSNR (dB)', 'PSNR（dB）'], higher: true, digits: 2,
    note: ['PSNR on foreground pixels against clean truth; methods train on noisy views.',
      '以無雜訊真值計算前景像素的 PSNR；各方法皆以含雜訊的視角訓練。']},
  sam: {i: 1, label: ['SAM (degrees)', 'SAM（度）'], higher: false, digits: 2,
    note: ['Mean spectral angle on foreground pixels; a lower SAM does not by itself imply better identification.',
      '前景像素的平均光譜角；較低的 SAM 本身並不代表較佳的辨識。']},
  miou: {i: 2, label: ['mIoU', 'mIoU'], higher: true, digits: 2,
    note: ['Dominant-endmember mIoU on test views. Each endmember is assigned to the class holding most of the training pixels it dominates, and this map is then frozen. No material label enters training; validation labels only select K.',
      '測試視角上以主導端元計算的 mIoU。每個端元指派給其主導之訓練像素中占多數的類別，之後凍結此對應關係。訓練不使用任何材質標籤；驗證標籤僅用於選擇 K。']},
  em: {i: 3, label: ['EM (degrees)', 'EM（度）'], higher: false, digits: 1,
    note: ['Mean angle between each class’s mean spectrum over pure training pixels and its learned endmember under one-to-one assignment.',
      '在一對一指派下，各類別於純訓練像素上的平均光譜與其學得端元之間的平均角度。']},
};

/* ---------- Shared helpers ---------- */
function tabKeys(e) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
  const tabs = $$('[role="tab"]', e.currentTarget);
  const i = tabs.indexOf(document.activeElement); if (i < 0) return;
  e.preventDefault();
  const n = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  tabs[n].focus(); tabs[n].click();
}
function selectTab(tabs, active) {
  for (const t of tabs) { const on = t === active; t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1; }
}
const press = (list, on) => list.forEach(b => b.setAttribute('aria-pressed', String(b === on)));

/* ---------- Hero loop ---------- */
/* hero.mp4: five 3.4 s segments with 0.5 s crossfades (scratchpad make_hero.py); captions follow the segment. */
(function hero() {
  const v = $('#hero-video'), btn = $('#hero-play'), cap = $('#hero-cap-text');
  if (!v) return;
  const lite = innerWidth < 861 || saveData;
  v.poster = lite ? v.dataset.litePoster : v.dataset.poster;
  const SEG = [
    ['Forbidden City', 'yellow roof tile abundance', 'yellow roof tile 的豐度'],
    ['Market', 'fired clay abundance', 'fired clay 的豐度'],
    ['City', 'living-leaf spectrum query', '活體葉片光譜查詢'],
    ['Karst', 'water abundance', 'water 的豐度'],
    ['Study', 'paper abundance', 'paper 的豐度']];
  const STEP = 3.4 - 0.5;
  let seg = -1;
  const show = i => { if (i === seg) return; seg = i; const s = SEG[i];
    cap.innerHTML = `<b>${s[0]}</b> · ${L('left: RGB rendering · right: ' + s[1], '左：RGB 渲染 · 右：' + s[2])}`; };
  show(0);
  v.addEventListener('timeupdate', () => show(Math.min(SEG.length - 1, Math.floor((v.currentTime + 0.25) / STEP))));
  const sync = () => {
    const p = v.paused;
    btn.querySelector('.lbl').innerHTML = p ? L('Play', '播放') : L('Pause', '暫停');
    btn.querySelector('svg').innerHTML = p ? '<path d="M3 1.5v9l7.5-4.5z"/>' : '<path d="M2 1h3v10H2zM7 1h3v10H7z"/>';
    btn.setAttribute('aria-label', p ? T('Play the background video', '播放背景影片') : T('Pause the background video', '暫停背景影片'));
  };
  v.addEventListener('play', sync); v.addEventListener('pause', sync);
  relabel.push(sync);
  let loaded = false;
  const load = () => { if (loaded) return; loaded = true; v.src = lite ? v.dataset.lite : v.dataset.src; };
  btn.addEventListener('click', () => { load(); if (v.paused) v.play().catch(() => {}); else v.pause(); });
  if (autoplayOK()) { load(); v.play().catch(() => {}); }
  sync();
  /* Pause when scrolled away. */
  if ('IntersectionObserver' in window) new IntersectionObserver(es => { for (const e of es) {
    if (!e.isIntersecting) { if (!v.paused) { v.pause(); v.dataset.auto = '1'; } }
    else if (v.dataset.auto) { delete v.dataset.auto; v.play().catch(() => {}); } } }).observe(v);
})();

/* ---------- Clip players ---------- */
const players = [];
function stopOthers(except) { for (const p of players) if (p !== except && !p.video.paused) p.video.pause(); }
document.addEventListener('visibilitychange', () => { if (document.hidden) for (const p of players) p.video.pause(); });
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
function setSource(video, src, keepTime, shouldPlay) {
  if (video.getAttribute('src') === src) return;
  const t = keepTime ? video.currentTime || 0 : 0;
  video.preload = 'auto';
  video.src = src;
  video.addEventListener('loadedmetadata', () => { if (t && t < video.duration) video.currentTime = t; }, {once: true});
  if (shouldPlay) video.play().catch(() => {});
}

/* ---------- Worlds grid ---------- */
(function worldsGrid() {
  const box = $('#worlds');
  for (const w of WORLDS) {
    const hasQuery = QUERY_WORLDS.some(q => q.key === w.key);
    const s = document.createElement('article');
    s.className = 'world';
    const set = SETTING[w.setting];
    s.innerHTML = `<img src="assets/worlds/${w.key}-400.webp" srcset="assets/worlds/${w.key}-400.webp 400w, assets/worlds/${w.key}.webp 800w" sizes="(max-width:980px) 50vw, 290px" width="800" height="450" loading="lazy" decoding="async" alt="Rendered view of the ${w.name} world">
      <div><h3>${w.name}</h3><span class="meta">${L(`${set[0]} · ${w.classes} classes (${w.rare} rare) · ${w.spectra} specimens`, `${set[1]} · ${w.classes} 類（${w.rare} 類稀有）· ${w.spectra} 種樣本`)}</span>
      <button type="button"${hasQuery ? '' : ' disabled'}>${hasQuery ? L('Query this world →', '查詢此世界 →') : L('Shown in the figures below', '見下方圖表')}</button></div>`;
    if (hasQuery) s.querySelector('button').addEventListener('click', () => {
      setQueryWorld(w.key);
      $('#query').scrollIntoView({behavior: reduceMotion.matches ? 'auto' : 'smooth'});
      const b = $(`#query-worlds button[data-world="${w.key}"]`); if (b) b.focus({preventScroll: true});
    });
    box.append(s);
  }
})();

/* ---------- Overview video (YouTube, click to load) ---------- */
$('#promo-play').addEventListener('click', () => {
  const promo = $('#promo'), f = document.createElement('iframe');
  f.src = 'https://www.youtube-nocookie.com/embed/' + promo.dataset.yt + '?autoplay=1&rel=0&modestbranding=1&playsinline=1' + (lang() === 'zh' ? '&hl=zh-TW' : '');
  f.title = 'Spectral World Renderer overview video';
  f.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture';
  f.allowFullscreen = true;
  promo.replaceChildren(f);
  for (const p of players) { p.video.pause(); p.userPaused = true; }
  const hv = $('#hero-video'); if (hv && !hv.paused) hv.pause();
  f.focus();
});

/* ---------- Simulated observation tabs ---------- */
(function hsiTabs() {
  const tabs = $$('#hsi-tabs [role="tab"]');
  for (const t of tabs) t.addEventListener('click', () => {
    selectTab(tabs, t);
    const n = t.dataset.hsi, img = $('#hsi-img');
    img.srcset = `assets/fig/swr_hsi_${n}-800.webp 800w, assets/fig/swr_hsi_${n}-1200.webp 1200w`;
    img.src = `assets/fig/swr_hsi_${n}-1200.webp`;
    $('#hsi-zoom').dataset.full = `assets/fig/swr_hsi_${n}.webp`;
    img.alt = `Simulated hyperspectral observations of ${t.textContent.replace(/ · /g, ', ')}: narrow bands, a colour-infrared composite, apparent reflectance and clean and noisy 900 nm crops.`;
  });
  $('#hsi-tabs').addEventListener('keydown', tabKeys);
})();

/* ---------- RGB vs single band (stills from the query clips) ---------- */
(function bandCompare() {
  const cv = $('#band-cv'), inp = $('input', cv);
  const f = () => cv.style.setProperty('--p', inp.value + '%');
  inp.addEventListener('input', f); f();
  let w = 'city', b = 'band_850nm';
  const set = (img, name) => { img.srcset = `assets/band/${name}-800.webp 800w, assets/band/${name}.webp 1280w`; img.src = `assets/band/${name}-800.webp`; };
  const draw = () => {
    set($('#band-a'), `${w}_rgb`); set($('#band-b'), `${w}_${b}`);
    const nm = b === 'band_850nm' ? '847 nm' : '553 nm';
    $('#band-tag').textContent = nm;
    $('#band-a').alt = `${NAMES[w]}, RGB rendering from the learned spectra`;
    $('#band-b').alt = `${NAMES[w]}, single band near ${nm} rendered by SWR`;
  };
  const wb = $$('#band-world button'), bb = $$('#band-nm button');
  wb.forEach(x => x.addEventListener('click', () => { w = x.dataset.w; press(wb, x); draw(); }));
  bb.forEach(x => x.addEventListener('click', () => { b = x.dataset.b; press(bb, x); draw(); }));
})();

/* ---------- Material query with RGB comparison ---------- */
const qVideo = $('#query-video'), qRgb = $('#query-rgb'), qStatus = $('#query-status'), qPlayer = $('#query-player');
let qWorld = QUERY_WORLDS[0], qChip = liveChips(qWorld)[1];
const qp = makePlayer(qVideo, () => { if (qp.ready) return; qp.ready = true; applyQuerySource(false); });
qVideo.addEventListener('play', () => { if (!qRgb.hidden) qRgb.play().catch(() => {}); });
qVideo.addEventListener('pause', () => qRgb.pause());
qVideo.addEventListener('seeked', () => { if (!qRgb.hidden) { qRgbSeeking = true; qRgb.currentTime = qVideo.currentTime; } });
/* Keep the RGB layer running and aligned: restart it if it stalls, nudge the rate for small drift,
   and seek only for large offsets (one seek at a time). */
let qRgbSeeking = false;
qRgb.addEventListener('seeked', () => { qRgbSeeking = false; });
setInterval(() => {
  if (qRgb.hidden || qVideo.paused) { qRgb.playbackRate = 1; return; }
  if (qRgb.paused && qRgb.readyState >= 2) qRgb.play().catch(() => {});
  if (qRgb.readyState < 2 || qRgbSeeking || !qVideo.duration) return;
  const dur = qVideo.duration;
  let d = qRgb.currentTime - qVideo.currentTime;
  if (d > dur / 2) d -= dur; else if (d < -dur / 2) d += dur;
  if (Math.abs(d) > 0.4) { qRgbSeeking = true; qRgb.currentTime = qVideo.currentTime % (qRgb.duration || dur); qRgb.playbackRate = 1; }
  else qRgb.playbackRate = Math.abs(d) < 0.03 ? 1 : Math.min(1.1, Math.max(0.9, 1 - d * 2));
}, 200);
qVideo.addEventListener('waiting', () => { qStatus.hidden = false; });
qVideo.addEventListener('playing', () => { qStatus.hidden = true; });
qVideo.addEventListener('canplay', () => { qStatus.hidden = true; });
qVideo.addEventListener('error', () => { qStatus.hidden = false; qStatus.innerHTML = L('Clip unavailable', '片段無法載入'); });

const qColour = c => QCOL[c.col] || QCOL.teal;
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
    case 'band': return '<svg viewBox="0 0 24 24" aria-hidden="true"><text x="12" y="18" text-anchor="middle" font-size="19" font-weight="700" fill="currentColor" font-family="Manrope, system-ui, sans-serif">λ</text></svg>';
    case 'combo': return open + '<path d="M5 5.5 12 19 19 5.5" stroke-width="3"/></svg>';
    case 'rgb': return open + '<circle cx="9" cy="9.5" r="5"/><circle cx="15" cy="9.5" r="5"/><circle cx="12" cy="15" r="5"/></svg>';
    default: return open + '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>';
  }
}
const labelHTML = text => text.split('∨').map(esc).join(OR_SVG);
const spoken = c => c.spoken || c.label;

/* Click queries: a per-frame track (fractions of the frame, x1000) places the crosshair on the clicked surface point. */
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
  qXhair.toggleAttribute('hidden', !p);
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
    b.setAttribute('aria-label', w.name);
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
  for (const b of $$('#query-worlds button')) b.setAttribute('aria-pressed', String(b.dataset.world === qWorld.key));
  renderChips();
  lazyPoster(qVideo, `assets/query/${qWorld.key}_poster.webp`);
  lazyPoster(qRgb, `assets/query/${qWorld.key}_poster.webp`);
  setChip(qChip.id, false);
}
function renderChips() {
  const chips = liveChips(qWorld), box = $('#query-chips');
  box.replaceChildren();
  for (const mode of Object.keys(QMODES)) {
    const group = chips.filter(c => c.mode === mode);
    if (!group.length) continue;
    const g = document.createElement('div'), m = QMODES[mode];
    g.className = 'qgroup'; g.setAttribute('role', 'group');
    g.setAttribute('aria-label', m.tag[0] ? T(m.tag[0] + ' queries', m.tag[1] + '查詢') : T('Rendering', '渲染'));
    g.innerHTML = `<span class="mode-tag" aria-hidden="true">${m.tag[0] ? L(m.tag[0], m.tag[1]) : L('Rendering', '渲染')}</span>`;
    for (const c of group) {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.chip = c.id; b.className = 'qchip';
      b.style.setProperty('--qc', c.mode === 'rgb' ? '#FFFFFF' : qColour(c));
      b.innerHTML = `<span class="qi">${modeIcon(c)}</span><span class="qt">${labelHTML(c.label)}</span>`;
      if (c.mode !== 'rgb') b.setAttribute('aria-label', `${T(m.aria[0], m.aria[1])}: ${spoken(c)}`);
      b.setAttribute('aria-pressed', String(c === qChip));
      b.addEventListener('click', () => { qp.userPaused = false; setChip(c.id); });
      g.append(b);
    }
    box.append(g);
  }
}
function chipDesc(c) {
  if (c.desc) return L(c.desc[0], c.desc[1]);
  if (c.mode === 'rgb') return L('Visible composite near 630 / 550 / 460 nm, rendered from the learned spectra.', '由學得光譜渲染、約 630 / 550 / 460 nm 的可見光合成影像。');
  if (c.mode === 'name') { const n = QCOL_NAME[c.col] || QCOL_NAME.teal;
    return L(`${n[0]} intensity shows the abundance of ${c.label} on a fixed 0–1 scale; all other surfaces are shown in grey.`,
      `以${n[1]}的濃淡表示 ${c.label} 的豐度（固定的 0–1 刻度）；其他表面以灰色顯示。`); }
  return L(`Single band near ${c.nm} nm; both band views share one fixed exposure, so brightness is comparable between wavelengths.` + (c.nm > 800 ? ' In the near-infrared, vegetation reflects strongly, whereas most building materials do not.' : ''),
    `約 ${c.nm} nm 的單一波段；兩個波段視圖共用同一固定曝光，因此不同波長間的亮度可直接比較。` + (c.nm > 800 ? '在近紅外波段，植被反射強烈，而多數建築材料則否。' : ''));
}
function setChip(id, keepTime = true) {
  const chips = liveChips(qWorld);
  qChip = chips.find(c => c.id === id) || chips[0];
  for (const b of $$('#query-chips button')) b.setAttribute('aria-pressed', String(b.dataset.chip === qChip.id));
  const compare = qChip.mode !== 'rgb', col = qColour(qChip), m = QMODES[qChip.mode];
  qRgb.hidden = !compare; $('#query-divider').hidden = !compare; $('#query-split').hidden = !compare;
  $('#label-l').hidden = !compare; $('#label-r').hidden = !compare;
  $('#label-r').innerHTML = m.right[0] ? L(m.right[0], m.right[1]) : esc(qChip.label);
  if (!compare) qRgb.pause();
  qPlayer.style.setProperty('--qc', compare ? col : '#FFFFFF');
  $('#query-icon').innerHTML = modeIcon(qChip);
  $('#query-term').innerHTML = qChip.mode === 'rgb' ? 'rgb' : labelHTML(qChip.label);
  qTrack = null; qXhair.toggleAttribute('hidden', true);
  qSpec.hidden = qChip.mode !== 'spectrum';
  if (qChip.mode === 'spectrum' && qChip.spec) {
    $('path.curve', qSpec).setAttribute('d', specPath(qChip.spec, 96, 34, 4, 4));
    $('.ref', qSpec).textContent = qChip.ref || 'reference';
    $('.ax', qSpec).textContent = qChip.specNm ? `${qChip.specNm[0]}–${qChip.specNm[1]} nm` : '';
  }
  $('#query-desc').innerHTML = `<b>${qWorld.name}</b> · ${chipDesc(qChip)}`;
  ariaVideo();
  applyQuerySource(keepTime);
}
function ariaVideo() {
  const c = qChip;
  qVideo.setAttribute('aria-label', `${qWorld.name}: ${c.mode === 'rgb' ? T('RGB rendering', 'RGB 渲染')
    : c.mode === 'band' ? c.label + (c.overlay ? '' : T(' band', ' 波段'))
    : c.mode === 'name' ? T('abundance of ', '豐度：') + c.label : T(QMODES[c.mode].aria[0], QMODES[c.mode].aria[1]) + ', ' + spoken(c)}`);
}
relabel.push(() => { renderChips(); ariaVideo(); });
$('#query-split').addEventListener('input', e => qPlayer.style.setProperty('--split', e.target.value + '%'));
qPlayer.addEventListener('dblclick', () => togglePlayer(qp));
qPlayer.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'k') { e.preventDefault(); togglePlayer(qp); } });
$('#query-chips').addEventListener('click', () => { qp.load(); if (qVideo.paused) qVideo.play().catch(() => {}); });
function syncPP(btn, v) { const playing = !v.paused; btn.setAttribute('aria-label', playing ? T('Pause clip', '暫停片段') : T('Play clip', '播放片段')); btn.textContent = playing ? '❚❚' : '▶'; }
function togglePlayer(p) {
  p.load();
  if (p.video.paused) { p.userPaused = false; p.video.play().catch(() => {}); } else { p.userPaused = true; p.video.pause(); }
}
setQueryWorld('nvidia');

/* ---------- Table 2 (all metrics in one grid) ---------- */
const MORDER = ['miou', 'psnr', 'sam', 'em'];
const MNAME = {miou: 'mIoU', psnr: 'PSNR', sam: 'SAM', em: 'EM'};
const METHODS = [['UnMix-NeRF', 'UnMix'], ['3DGS + VCA / NNLS', '3DGS+VCA'], ['SWR', 'SWR']];
function renderTable() {
  const head = $('#bench-table thead'), body = $('#bench-table tbody');
  const top = `<tr><th scope="col" rowspan="2"><span class="en">World</span><span class="zh">世界</span></th>` +
    MORDER.map(k => `<th scope="colgroup" colspan="3" class="mg">${MNAME[k]} ${METRICS[k].higher ? '↑' : '↓'}</th>`).join('') + '</tr>';
  const sub = '<tr>' + MORDER.map(() => METHODS.map((n, j) =>
    `<th scope="col" class="${j === 2 ? 'ours ' : ''}${j === 0 ? 'gs' : ''}"><span class="long">${n[0]}</span><span class="short" aria-hidden="true">${n[1]}</span></th>`).join('')).join('') + '</tr>';
  head.innerHTML = top + sub;
  body.replaceChildren();
  const means = MORDER.map(() => [0, 0, 0]);
  const cells = (vals, k, mark) => {
    const m = METRICS[k], best = m.higher ? Math.max(...vals) : Math.min(...vals);
    return vals.map((v, j) => `<td class="${j === 2 ? 'ours ' : ''}${j === 0 ? 'gs ' : ''}${mark && v === best ? 'best' : ''}">${v.toFixed(m.digits)}</td>`).join('');
  };
  for (const w of WORLDS) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<th scope="row">${w.name}<span class="set">${L(...SETTING[w.setting])}</span></th>` +
      MORDER.map((k, a) => { const vals = TABLE2[w.key].map(r => r[METRICS[k].i]); vals.forEach((v, j) => means[a][j] += v / WORLDS.length); return cells(vals, k, true); }).join('');
    body.append(tr);
  }
  const tr = document.createElement('tr'); tr.className = 'mean';
  tr.innerHTML = `<th scope="row">${L('Mean', '平均')}</th>` + MORDER.map((k, a) => cells(means[a], k, false)).join('');
  body.append(tr);
  $('#bench-caption').innerHTML = L(
    'All four metrics on the eight worlds; the amber dot marks the best value per world and metric. mIoU: dominant-endmember mIoU on test views. PSNR (dB) and SAM (degrees) on foreground pixels against clean truth; a lower SAM does not by itself imply better identification. EM (degrees): angle between each class mean spectrum and its learned endmember. Source: Table 2 of the paper.',
    '八個世界上的四項指標；琥珀色圓點標示各世界、各指標的最佳數值。mIoU：測試視角上以主導端元計算的 mIoU。PSNR（dB）與 SAM（度）以無雜訊真值計算前景像素；較低的 SAM 本身並不代表較佳的辨識。EM（度）：各類別平均光譜與其學得端元之間的角度。來源：論文表 2。');
}
renderTable();

/* ---------- Efficiency bars (Table 6) ---------- */
function bars(el, items, max) {
  for (const it of items) {
    const w = 100 * it.v / max, inside = w > 70;
    const row = document.createElement('div');
    row.className = 'hbar' + (it.ours ? ' ours-row' : '');
    row.innerHTML = `<span class="name">${it.name}</span><span class="track2"><span class="fill" style="width:${w}%"></span><span class="val${inside ? ' inside' : ''}" style="--w:${w}%">${L(it.label[0], it.label[1])}</span></span>`;
    el.append(row);
  }
}
bars($('#train-bars'), [
  {name: 'SWR', v: 397, label: ['6 min 37 s', '6 分 37 秒'], ours: true},
  {name: 'UnMix-NeRF', v: 1457, label: ['24 min 17 s', '24 分 17 秒']},
  {name: '3DGS + VCA / NNLS', v: 5102, label: ['85 min 2 s', '85 分 2 秒']}], 5102);
bars($('#fps-bars'), [
  {name: 'SWR', v: 34.9, label: ['34.9 views/s', '每秒 34.9 個視角'], ours: true},
  {name: '3DGS + VCA / NNLS', v: 3.6, label: ['3.6 views/s', '每秒 3.6 個視角']},
  {name: 'UnMix-NeRF', v: 0.18, label: ['0.18 views/s', '每秒 0.18 個視角']}], 34.9);

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
  selectTab($$('#tetra-worlds [role="tab"]'), $(`#tetra-worlds [data-world="${k}"]`));
  lazyPoster(tVideo, `assets/tetra/${k}.webp`);
  if (tp.ready) setSource(tVideo, `assets/tetra/${k}.mp4`, false, tp.visible && autoplayOK() && !tp.userPaused);
}
tVideo.addEventListener('click', () => togglePlayer(tp));
setTetra(tKey);

/* Visible play/pause buttons (WCAG 2.2.2). */
for (const [btn, p] of [[$('#query-pp'), qp], [$('#tetra-pp'), tp]]) {
  btn.addEventListener('click', e => { e.stopPropagation(); togglePlayer(p); });
  for (const ev of ['play', 'pause']) p.video.addEventListener(ev, () => syncPP(btn, p.video));
  relabel.push(() => syncPP(btn, p.video));
}

/* ---------- Lightbox ---------- */
const box = $('#lightbox');
for (const z of $$('.fig')) z.addEventListener('click', () => {
  const img = $('img', box);
  img.src = z.dataset.full; img.alt = $('img', z).alt;
  if (box.showModal) box.showModal(); else window.open(z.dataset.full, '_blank', 'noopener');
});
$('.close', box).addEventListener('click', () => box.close());
box.addEventListener('click', e => { if (e.target === box) box.close(); });

/* ---------- BibTeX copy ---------- */
$('#copy-bib').addEventListener('click', async () => {
  const text = $('#bibtex').textContent, status = $('#copy-status');
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else { const a = document.createElement('textarea'); a.value = text; a.style.position = 'fixed'; a.style.opacity = '0'; document.body.append(a); a.select(); const ok = document.execCommand('copy'); a.remove(); if (!ok) throw Error(); }
    status.innerHTML = L('Copied', '已複製');
  } catch {
    const r = document.createRange(); r.selectNodeContents($('#bibtex')); const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    status.innerHTML = L('Selected; copy it with the keyboard.', '已選取，請以鍵盤複製。');
  }
  setTimeout(() => { status.textContent = ''; }, 2500);
});

setLang(lang(), false);

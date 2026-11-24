// ---------------------------------------------------------
// SHARED STATE AND SMALL UI HELPERS
// ---------------------------------------------------------
const $ = (id) => document.getElementById(id);
const state = {
  token: sessionStorage.getItem('fridge-token'), name: sessionStorage.getItem('fridge-name'),
  library: null, current: null, detail: null,
  selected: null, signature: '',
};
// Samma fasta mått som i program.hpp.
const IGNORE_RIGHT_PIXELS = 450, BORDER_ZONE_PIXELS = 100;

async function api(path, method = 'GET', body = undefined) {
  const headers = {};
  if (state.token) headers['X-Session'] = state.token;
  if (body !== undefined && !(body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(body);
  }
  const response = await fetch(path, {method, headers, body});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}).`);
    error.status = response.status;
    if (response.status === 401) join();
    throw error;
  }
  return response.json();
}

function toast(message, error = false) {
  $('toast').textContent = message;
  $('toast').className = `toast${error ? ' error' : ''}`;
  $('toast').hidden = false;
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => { $('toast').hidden = true; }, error ? 10000 : 5000);
}

function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

const seconds = (value) => `${value.toFixed(1)} s`;

// ---------------------------------------------------------
// JOIN WITH A NAME
// ---------------------------------------------------------
function join() {
  if (!$('join-dialog').open) $('join-dialog').showModal();
}

$('join-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const result = await api('/api/editors', 'POST', {name: $('editor-name').value});
  state.token = result.token; state.name = result.name;
  sessionStorage.setItem('fridge-token', result.token);
  sessionStorage.setItem('fridge-name', result.name);
  $('user-badge').textContent = result.name;
  $('join-dialog').close();
});

// ---------------------------------------------------------
// CLIP LIST, POLLED SO PROGRESS AND OTHER USERS' CLIPS APPEAR
// ---------------------------------------------------------
async function refresh() {
  try {
    state.library = await api('/api/videos');
  } catch (error) {
    toast(error.message, true);
    return;
  }
  const library = state.library;
  if (!$('upload-camera').options.length) {
    for (const camera of Object.keys(library.cameras)) $('upload-camera').append(new Option(camera, camera));
  }
  const signature = JSON.stringify(library.items.map((item) => [item.id, item.status, item.progress, item.message, item.processing, item.reviewed, item.movements]));
  if (signature !== state.signature) {
    state.signature = signature;
    renderList();
    const current = library.items.find((item) => item.id === state.current);
    if (current && current.status === 'done' && (!state.detail || state.detail.meta.status !== 'done')) openVideo(current.id);
    else if (current && state.detail) { state.detail.meta = current; renderHeading(); }
  }
}

function renderList() {
  const list = $('video-list');
  list.replaceChildren();
  if (!state.library.items.length) list.append(element('p', {class: 'queue-empty'}, 'No clips yet.'));
  for (const item of state.library.items) {
    const status = item.status === 'done' ? `${item.movements ?? 0} movements` : item.status === 'failed' ? 'failed' : item.processing ? `${item.processing.stage}: ${item.processing.message}` : item.message;
    const card = element('button', {class: `image-card${item.id === state.current ? ' active' : ''}`, onclick: () => openVideo(item.id)},
      element('div', {class: 'card-info'},
        element('div', {class: 'card-top'}, element('strong', {}, item.filename), element('span', {}, item.camera)),
        element('div', {class: 'card-bottom'}, `${status}${item.reviewed ? ', reviewed' : ''}, ${new Date(item.created * 1000).toLocaleString()}`)));
    if (item.status === 'running' || item.status === 'queued') {
      const progress = item.processing;
      const percent = progress?.total > 0 ? 100 * progress.completed / progress.total : item.progress;
      card.append(element('div', {class: 'card-progress', style: `width:${percent}%`}));
    }
    list.append(card);
  }
}

// ---------------------------------------------------------
// OPEN ONE CLIP
// ---------------------------------------------------------
async function openVideo(id) {
  state.current = id;
  state.selected = null;
  state.detail = await api(`/api/videos/${id}`);
  renderList();
  renderHeading();
  const ready = state.detail.meta.status === 'done' && state.detail.result;
  $('video-empty').hidden = ready;
  $('video-frame').hidden = !ready;
  if (ready) {
    const source = `/api/videos/${id}/files/video.mp4`;
    if (!$('player').src.endsWith(source)) $('player').src = source;
  }
  renderInspector();
  draw();
}

function renderHeading() {
  const {meta, result} = state.detail;
  $('video-title').textContent = meta.filename;
  const info = result ? `, ${seconds(result.video.duration)}, ${result.video.width}×${result.video.height}` : '';
  $('video-meta').textContent = `${meta.camera} camera, event ${meta.event_id}${info}`;
  const progress = meta.processing;
  $('video-state').textContent = meta.status === 'done' ? 'Ready' : progress?.stage || meta.status;
  $('processing-title').textContent = meta.status === 'failed' ? 'Körningen misslyckades' : meta.status === 'queued' ? 'Väntar i kö' : progress?.stage || 'Förbereder körningen';
  $('processing-detail').textContent = progress?.message || meta.message;
  $('processing-progress').hidden = meta.status !== 'running';
  if (progress?.total > 0) $('processing-progress').value = 100 * progress.completed / progress.total;
  else $('processing-progress').removeAttribute('value');
  const warnings = result ? result.warnings : [];
  $('video-warning').hidden = !warnings.length;
  $('video-warning').textContent = warnings.join(' ');
  $('ignored-area-info').textContent = `Högra ${IGNORE_RIGHT_PIXELS} px ignoreras. De ${BORDER_ZONE_PIXELS} px direkt till vänster är gränszonen och räknas som inne.`;
}

// ---------------------------------------------------------
// MOVEMENTS AND TRACK CORRECTIONS
// ---------------------------------------------------------
function review(trackId) {
  const saved = state.detail.review?.tracks?.[trackId];
  const track = state.detail.result.tracks.find((row) => row.id === trackId);
  return saved || {category: track.category.startsWith('imagenet-') ? 'unknown' : track.category, ignored: track.ignored, direction: 'auto'};
}

// Visar modellens tre bästa förslag i den ordning som sparats i resultatet.
function modelPredictions(title, predictions) {
  const section = element('div', {class: 'model-predictions'}, element('strong', {}, title));
  for (const row of (predictions || []).slice(0, 3)) {
    section.append(element('div', {class: 'model-prediction'}, element('span', {}, row.category.replace('imagenet-', '')), element('span', {class: 'prediction-score'}, `${(row.confidence * 100).toFixed(1)}%`)));
  }
  if (!predictions?.length) section.append(element('div', {class: 'muted'}, 'No saved predictions'));
  return section;
}

function renderInspector() {
  const result = state.detail?.result;
  $('movement-list').replaceChildren();
  $('track-list').replaceChildren();
  $('notes').value = state.detail?.review?.notes || '';
  $('reviewed').checked = !!state.detail?.review?.reviewed;
  if (!result) { $('movement-count').textContent = '0'; $('track-count').textContent = '0'; return; }
  $('movement-count').textContent = result.movements.length;
  if (!result.movements.length) $('movement-list').append(element('p', {class: 'muted'}, 'No in/out movement detected.'));
  for (const movement of result.movements) {
    // Movement times are door-event times; the player shows this clip, which starts at the offset.
    const clipTime = Math.max(0, movement.start - (result.offset_seconds || 0));
    $('movement-list').append(element('div', {class: `movement-row ${movement.direction}`, onclick: () => seek(movement.track_id, clipTime)},
      element('b', {}, movement.direction.toUpperCase()), element('span', {}, `#${movement.track_id} ${movement.category}`),
      element('span', {class: 'muted'}, `${seconds(movement.start)}${movement.classification_confidence != null ? `, ${(movement.classification_confidence * 100).toFixed(0)}%` : ''}`)));
  }
  // Moving or classified tracks first; static clutter last.
  const tracks = [...result.tracks].sort((a, b) => (b.events.length - a.events.length) || ((b.movement || 0) - (a.movement || 0)));
  $('track-count').textContent = tracks.length;
  for (const track of tracks) {
    const values = review(track.id);
    const category = element('select', {'data-track': track.id, 'data-field': 'category'});
    category.append(new Option('unknown', 'unknown'));
    for (const name of state.library.categories) category.append(new Option(name, name));
    category.value = values.category;
    const direction = element('select', {'data-track': track.id, 'data-field': 'direction'});
    for (const [value, label] of [['auto', `auto (${track.events.map((e) => e.direction).join(', ') || 'none'})`], ['in', 'in'], ['out', 'out'], ['none', 'no movement']]) direction.append(new Option(label, value));
    direction.value = values.direction;
    const ignore = element('input', {type: 'checkbox', 'data-track': track.id, 'data-field': 'ignored'});
    ignore.checked = values.ignored;
    // Äldre resultat med enbart ViT har sina förslag i top_categories.
    const vitPredictions = track.vit_top_categories || (track.label_source === 'grocery-vit' ? track.top_categories : []);
    const decision = track.classification_confidence == null ? 'Not classified' : `${track.category} ${(track.classification_confidence * 100).toFixed(1)}%`;
    const start = track.start ?? track.observations[0].time, end = track.end ?? track.observations.at(-1).time;
    const image = track.crop_file ? element('img', {src: `/api/videos/${state.current}/files/${track.crop_file}`, alt: `Object ${track.id}`, onclick: () => seek(track.id, track.time ?? start)}) : element('div');
    $('track-list').append(element('div', {class: `track-row${state.selected === track.id ? ' active' : ''}`, id: `track-${track.id}`}, image,
      element('div', {class: 'object-details'},
        element('strong', {}, `#${track.id}`),
        element('div', {class: 'muted'}, `${seconds(start)}–${seconds(end)}`),
        element('div', {class: 'muted'}, track.inside == null ? '' : `Slutläge: ${track.inside ? 'Inne' : 'Ute'}`),
        modelPredictions('ViT top 3', vitPredictions),
        modelPredictions('YOLO top 3', track.yolo_top_categories),
        element('div', {class: 'object-decision'}, element('strong', {}, 'Decision'), element('div', {}, decision)),
        category, direction, element('label', {class: 'ignore'}, ignore, 'Ignore (not a product)'))));
  }
}

function seek(trackId, time) {
  state.selected = trackId;
  $('player').pause();
  $('player').currentTime = time;
  document.querySelectorAll('.track-row').forEach((row) => row.classList.toggle('active', row.id === `track-${trackId}`));
  $(`track-${trackId}`)?.scrollIntoView({block: 'nearest'});
  draw();
}

$('save-review').addEventListener('click', async () => {
  if (!state.detail?.result) return;
  const tracks = {};
  for (const track of state.detail.result.tracks) tracks[track.id] = review(track.id);
  for (const input of document.querySelectorAll('[data-track]')) {
    tracks[input.dataset.track][input.dataset.field] = input.type === 'checkbox' ? input.checked : input.value;
  }
  try {
    state.detail.review = await api(`/api/videos/${state.current}/review`, 'PUT', {tracks, notes: $('notes').value, reviewed: $('reviewed').checked});
    toast('Review saved.');
    state.signature = '';
    refresh();
  } catch (error) { toast(error.message, true); }
});

// Visar samma fasta område som programmet hoppar över.
function drawIgnoredArea(context, pictureBox) {
  const result = state.detail?.result;
  if (!result) return;
  const ignored = {box: [Math.max(0, (result.video.width - IGNORE_RIGHT_PIXELS) / result.video.width), 0, 1, 1]};
  const [x, y, width, height] = pictureBox(ignored.box);
  context.save();
  context.beginPath();
  context.rect(x, y, width, height);
  context.clip();
  context.fillStyle = 'rgba(180, 35, 35, 0.20)';
  context.fillRect(x, y, width, height);
  context.strokeStyle = 'rgba(160, 35, 35, 0.45)';
  for (let line = x - height; line < x + width; line += 16) {
    context.beginPath();
    context.moveTo(line, y);
    context.lineTo(line + height, y + height);
    context.stroke();
  }
  context.fillStyle = '#812020';
  context.fillText('Inne: ignoreras av YOLO', x + 8, y + 20);
  context.restore();
}

// ---------------------------------------------------------
// OVERLAY: ZONES AND THE BOXES VISIBLE AT THE CURRENT TIME
// ---------------------------------------------------------
function draw() {
  const canvas = $('overlay'), video = $('player');
  if ($('video-frame').hidden || !video.videoWidth) return;
  const width = video.clientWidth, height = video.clientHeight - 40;  // Leave the controls usable.
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  // Map normalized coordinates onto the letterboxed picture inside the <video> element.
  const scale = Math.min(width / video.videoWidth, video.clientHeight / video.videoHeight);
  const pictureWidth = video.videoWidth * scale, pictureHeight = video.videoHeight * scale;
  const left = (width - pictureWidth) / 2, top = (video.clientHeight - pictureHeight) / 2;
  const box = ([x1, y1, x2, y2]) => [left + x1 * pictureWidth, top + y1 * pictureHeight, (x2 - x1) * pictureWidth, (y2 - y1) * pictureHeight];
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, width, height);
  context.lineWidth = 2;
  context.font = '12px Arial';
  drawIgnoredArea(context, box);
  const result = state.detail?.result;
  if (!result) return;
  const originalWidth = result.video.width;
  const border = Math.max(0, (originalWidth - IGNORE_RIGHT_PIXELS - BORDER_ZONE_PIXELS) / originalWidth);
  const inside = Math.max(0, (originalWidth - IGNORE_RIGHT_PIXELS) / originalWidth);
  context.fillStyle = 'rgba(224, 123, 32, 0.12)';
  context.fillRect(...box([0, 0, border, 1]));
  context.fillStyle = 'rgba(43, 154, 72, 0.20)';
  context.fillRect(...box([border, 0, inside, 1]));
  context.fillStyle = '#222';
  context.fillText('Ute', left + 4, top + 14);
  context.fillText('Gränszon = inne', left + border * pictureWidth + 4, top + 14);
  const window = 1 / (result.sample_fps || result.video.fps || 25);
  const now = video.currentTime;
  for (const track of result.tracks) {
    if (!$('show-all').checked && !track.events.length && track.classification_confidence == null && track.id !== state.selected) continue;
    const nearest = track.box ? track : track.observations.reduce((best, row) => Math.abs(row.time - now) < Math.abs(best.time - now) ? row : best);
    if (Math.abs(nearest.time - now) > window) continue;
    const values = review(track.id);
    if (values.ignored) continue;
    context.strokeStyle = track.id === state.selected ? '#ffd400' : (track.events.length ? '#2f7fd8' : '#9aa');
    context.strokeRect(...box(nearest.box));
    const label = `#${track.id} ${values.category !== 'unknown' ? values.category : track.category.replace('imagenet-', '~')}`;
    const [x, y] = box(nearest.box);
    context.fillStyle = '#000a';
    context.fillRect(x, y - 16, context.measureText(label).width + 8, 16);
    context.fillStyle = 'white';
    context.fillText(label, x + 4, y - 4);
  }
}

$('player').addEventListener('timeupdate', draw);
$('player').addEventListener('seeked', draw);
$('player').addEventListener('loadedmetadata', draw);
$('show-all').addEventListener('change', draw);
window.addEventListener('resize', draw);
(function animate() { if (!$('player').paused) draw(); requestAnimationFrame(animate); })();

$('rerun-button').addEventListener('click', async () => {
  if (!state.current) return;
  try { await api(`/api/videos/${state.current}/rerun`, 'POST'); state.detail.meta.status = 'queued'; openVideo(state.current); refresh(); }
  catch (error) { toast(error.message, true); }
});

// ---------------------------------------------------------
// EXPORT REVIEWED TRACK CROPS FOR VIT TRAINING
// ---------------------------------------------------------
$('export-button').addEventListener('click', async () => {
  const response = await fetch('/api/videos/export', {headers: {'X-Session': state.token || ''}});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    toast(payload.detail || 'Export failed.', true);
    if (response.status === 401) join();
    return;
  }
  const link = element('a', {href: URL.createObjectURL(await response.blob()), download: 'fridge-video-crops.zip'});
  link.click();
  URL.revokeObjectURL(link.href);
});

// ---------------------------------------------------------
// UPLOAD
// ---------------------------------------------------------
$('upload-button').addEventListener('click', () => { $('upload-result').textContent = ''; $('upload-dialog').showModal(); });
$('cancel-upload').addEventListener('click', () => $('upload-dialog').close());
$('upload-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = new FormData();
  form.append('file', $('upload-file').files[0]);
  form.append('camera', $('upload-camera').value);
  form.append('event_id', $('upload-event').value);
  $('upload-result').textContent = 'Uploading…';
  try {
    const result = await api('/api/videos', 'POST', form);
    state.current = result.id;
    $('upload-dialog').close();
    toast('Uploaded. Processing…');
    state.signature = '';
    await refresh();
    openVideo(result.id);
  } catch (error) { $('upload-result').textContent = error.message; }
});

// ---------------------------------------------------------
// START
// ---------------------------------------------------------
if (state.token) $('user-badge').textContent = state.name; else join();
refresh();
setInterval(refresh, 2000);

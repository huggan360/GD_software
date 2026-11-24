// ---------------------------------------------------------
// SHARED STATE AND SMALL UI HELPERS
// ---------------------------------------------------------
const $ = (id) => document.getElementById(id);
const state = {
  token: sessionStorage.getItem('fridge-token'), name: sessionStorage.getItem('fridge-name'),
  library: null, signature: '', current: null, detail: null, picture: null,
  boxes: [], selected: null, dirty: false, drag: null, scale: 1,
  category: sessionStorage.getItem('fridge-image-category') || '',
};

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
  toast.timeout = setTimeout(() => { $('toast').hidden = true; }, error ? 10000 : 4000);
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

// Varje vara får en egen, stabil färg.
function color(category) {
  let hash = 0;
  for (const letter of category) hash = (hash * 31 + letter.charCodeAt(0)) % 360;
  return `hsl(${hash}, 85%, 45%)`;
}

// ---------------------------------------------------------
// JOIN WITH A NAME (SAME SESSION AS THE VIDEO PAGE)
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
// IMAGE LIST, POLLED SO OTHER PEOPLE'S WORK APPEARS
// ---------------------------------------------------------
async function refresh() {
  try {
    state.library = await api('/api/images');
  } catch (error) { toast(error.message, true); return; }
  const library = state.library;
  // Fridge left acquisition: the same address now shows the video viewer.
  if (library.mode !== 'acquisition') { if (!state.dirty) location.reload(); return; }
  $('take-photo').disabled = !library.camera.available;
  $('take-photo').title = library.camera.reason;
  $('camera-warning').hidden = library.camera.available;
  $('camera-warning').textContent = library.camera.reason;
  $('image-progress').textContent = `${library.annotated} / ${library.items.length} done`;
  $('dataset-count').textContent = `${library.annotated} annotated`;
  renderCategories();
  const signature = JSON.stringify([library.items.map((item) => [item.id, item.annotated, item.boxes]), $('only-open').checked]);
  if (signature !== state.signature) { state.signature = signature; renderList(); }
}

function renderCategories() {
  const select = $('current-category');
  const names = state.library.categories;
  if (select.options.length !== names.length) {
    select.replaceChildren(...names.map((name) => new Option(name, name)));
  }
  if (names.includes(state.category)) select.value = state.category;
  state.category = select.value;
}

$('current-category').addEventListener('change', () => {
  state.category = $('current-category').value;
  sessionStorage.setItem('fridge-image-category', state.category);
  // Ändrar även den markerade rutan, så att man kan rätta en ruta direkt.
  if (state.selected != null) { state.boxes[state.selected].category = state.category; changed(); }
});

$('new-category').addEventListener('click', async () => {
  const name = prompt('New item type (lower case, for example oat-milk):');
  if (!name) return;
  try {
    const result = await api('/api/categories', 'POST', {name: name.trim()});
    state.category = result.name;
    $('current-category').replaceChildren();
    await refresh();
    toast(`Added ${result.name}.`);
  } catch (error) { toast(error.message, true); }
});

function visibleItems() {
  return state.library.items.filter((item) => !$('only-open').checked || !item.annotated || item.id === state.current);
}

function renderList() {
  const list = $('image-list');
  list.replaceChildren();
  const items = visibleItems();
  if (!items.length) list.append(element('p', {class: 'queue-empty'}, state.library.items.length ? 'Everything is annotated.' : 'No images yet.'));
  for (const item of items) {
    const label = item.annotated ? (item.boxes ? `✓ ${item.boxes} box${item.boxes === 1 ? '' : 'es'}` : '✓ empty') : 'Not annotated';
    list.append(element('button', {class: `thumb-card${item.annotated ? ' annotated' : ''}${item.id === state.current ? ' active' : ''}`, onclick: () => openImage(item.id)},
      element('img', {src: `/api/images/${item.id}/thumb`, alt: item.filename, loading: 'lazy'}),
      element('div', {class: 'thumb-label'}, element('span', {}, label), element('span', {class: 'muted'}, new Date(item.created * 1000).toLocaleTimeString()))));
  }
}

$('only-open').addEventListener('change', refresh);

// ---------------------------------------------------------
// OPEN ONE IMAGE
// ---------------------------------------------------------
async function openImage(id) {
  if (state.dirty && id !== state.current && !confirm('Discard the unsaved boxes in this image?')) return;
  try { state.detail = await api(`/api/images/${id}`); } catch (error) { toast(error.message, true); return; }
  state.current = id;
  state.boxes = (state.detail.annotation?.boxes || []).map((row) => ({category: row.category, box: [...row.box]}));
  state.selected = null;
  state.dirty = false;
  const picture = new Image();
  picture.onload = () => { if (state.current === id) { state.picture = picture; layout(); } };
  picture.src = `/api/images/${id}/file`;
  const {meta} = state.detail;
  $('image-title').textContent = meta.filename;
  $('image-meta').textContent = `${meta.width}×${meta.height}, ${meta.source} by ${meta.editor}, ${new Date(meta.created * 1000).toLocaleString()}`;
  $('thermal-panel').hidden = !meta.thermal;
  if (meta.thermal) {
    $('thermal-image').src = `/api/images/${id}/thermal`;
    $('thermal-range').textContent = `${meta.thermal.min_c}–${meta.thermal.max_c} °C`;
  }
  $('image-empty').hidden = true;
  $('image-canvas').hidden = false;
  renderStatus();
  renderBoxes();
  renderList();
}

function renderStatus() {
  const annotation = state.detail?.annotation;
  $('image-state').textContent = state.dirty ? 'Unsaved changes' : annotation ? 'Annotated' : 'Not annotated';
  $('image-state').className = `save-state ${state.dirty ? 'unlabelled' : annotation ? 'saved' : ''}`;
  $('annotation-info').textContent = annotation ? `Saved by ${annotation.editor}, ${new Date(annotation.saved_at * 1000).toLocaleString()}` : '';
}

function changed() {
  state.dirty = true;
  renderStatus();
  renderBoxes();
  draw();
}

function renderBoxes() {
  const list = $('box-list');
  list.replaceChildren();
  $('box-count').textContent = state.boxes.length;
  state.boxes.forEach((row, index) => {
    const select = element('select', {onchange: (event) => { row.category = event.target.value; changed(); }, onclick: (event) => event.stopPropagation()});
    for (const name of state.library?.categories || []) select.append(new Option(name, name));
    select.value = row.category;
    list.append(element('div', {class: `box-row${index === state.selected ? ' active' : ''}`, onclick: () => { state.selected = index; renderBoxes(); draw(); }},
      element('span', {class: 'box-swatch', style: `background:${color(row.category)}`}),
      element('span', {class: 'box-number'}, `#${index + 1}`), select,
      element('button', {class: 'danger-text', title: 'Delete box', onclick: (event) => { event.stopPropagation(); removeBox(index); }}, '✕')));
  });
}

function removeBox(index) {
  state.boxes.splice(index, 1);
  state.selected = null;
  changed();
}

// ---------------------------------------------------------
// CANVAS: DRAW, SELECT, MOVE AND RESIZE BOXES (IMAGE PIXELS)
// ---------------------------------------------------------
function layout() {
  const canvas = $('image-canvas'), stage = $('image-stage'), picture = state.picture;
  if (!picture) return;
  state.scale = Math.min((stage.clientWidth - 24) / picture.width, (stage.clientHeight - 24) / picture.height);
  canvas.width = Math.round(picture.width * state.scale);
  canvas.height = Math.round(picture.height * state.scale);
  draw();
}

function draw() {
  const canvas = $('image-canvas'), context = canvas.getContext('2d');
  if (!state.picture) return;
  const s = state.scale;
  context.drawImage(state.picture, 0, 0, canvas.width, canvas.height);
  context.font = '13px Arial';
  state.boxes.forEach((row, index) => {
    const [x1, y1, x2, y2] = row.box.map((value) => value * s);
    const selected = index === state.selected;
    context.lineWidth = selected ? 3 : 2;
    context.strokeStyle = color(row.category);
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `#${index + 1} ${row.category}`;
    context.fillStyle = color(row.category);
    context.fillRect(x1, Math.max(0, y1 - 18), context.measureText(label).width + 8, 18);
    context.fillStyle = 'white';
    context.fillText(label, x1 + 4, Math.max(13, y1 - 5));
    if (selected) {
      context.fillStyle = 'white';
      for (const [x, y] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) { context.fillRect(x - 4, y - 4, 8, 8); context.strokeRect(x - 4, y - 4, 8, 8); }
    }
  });
}

function point(event) {
  const rect = $('image-canvas').getBoundingClientRect();
  const clamp = (value, limit) => Math.max(0, Math.min(limit, value));
  return [clamp((event.clientX - rect.left) / state.scale, state.picture.width), clamp((event.clientY - rect.top) / state.scale, state.picture.height)];
}

$('image-canvas').addEventListener('pointerdown', (event) => {
  if (!state.picture) return;
  const [x, y] = point(event);
  const near = 8 / state.scale;
  $('image-canvas').setPointerCapture(event.pointerId);
  // 1: ett hörn på den markerade rutan ändrar storlek.
  if (state.selected != null) {
    const box = state.boxes[state.selected].box;
    for (const [cx, cy] of [[0, 1], [2, 1], [0, 3], [2, 3]]) {
      if (Math.abs(box[cx] - x) < near && Math.abs(box[cy] - y) < near) {
        state.drag = {mode: 'resize', index: state.selected, fixed: [box[2 - cx], box[4 - cy]]};
        return;
      }
    }
  }
  // 2: klick i en ruta markerar och flyttar den (minsta rutan först, så att små rutor går att nå).
  const hits = state.boxes.map((row, index) => [row, index]).filter(([row]) => x >= row.box[0] && x <= row.box[2] && y >= row.box[1] && y <= row.box[3]);
  hits.sort(([a], [b]) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
  if (hits.length) {
    const [row, index] = hits[0];
    state.selected = index;
    state.drag = {mode: 'move', index, start: [x, y], original: [...row.box]};
    renderBoxes(); draw();
    return;
  }
  // 3: annars ritas en ny ruta med den valda varan.
  if (!state.category) { toast('Add an item type first (+ Add a new item type).', true); return; }
  state.boxes.push({category: state.category, box: [x, y, x, y]});
  state.selected = state.boxes.length - 1;
  state.drag = {mode: 'resize', index: state.selected, fixed: [x, y], fresh: true};
  draw();
});

$('image-canvas').addEventListener('pointermove', (event) => {
  const drag = state.drag;
  if (!drag) return;
  const [x, y] = point(event);
  const row = state.boxes[drag.index];
  if (drag.mode === 'resize') {
    row.box = [Math.min(x, drag.fixed[0]), Math.min(y, drag.fixed[1]), Math.max(x, drag.fixed[0]), Math.max(y, drag.fixed[1])];
  } else {
    const [x1, y1, x2, y2] = drag.original;
    const dx = Math.max(-x1, Math.min(state.picture.width - x2, x - drag.start[0]));
    const dy = Math.max(-y1, Math.min(state.picture.height - y2, y - drag.start[1]));
    row.box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy];
  }
  drag.moved = true;
  draw();
});

$('image-canvas').addEventListener('pointerup', () => {
  const drag = state.drag;
  state.drag = null;
  if (!drag) return;
  const row = state.boxes[drag.index];
  // En ny ruta som bara är ett klick räknas inte.
  if (drag.fresh && ((row.box[2] - row.box[0]) * state.scale < 5 || (row.box[3] - row.box[1]) * state.scale < 5)) {
    state.boxes.splice(drag.index, 1);
    state.selected = null;
    renderBoxes(); draw();
    return;
  }
  if (drag.moved) changed(); else { renderBoxes(); draw(); }
});

window.addEventListener('resize', layout);

// ---------------------------------------------------------
// SAVE, DELETE AND NAVIGATE
// ---------------------------------------------------------
async function save() {
  if (!state.current) return;
  if (!state.boxes.length && !confirm('No boxes: save this image as empty (no products)?')) return;
  try {
    const boxes = state.boxes.map((row) => ({category: row.category, box: row.box.map(Math.round)}));
    state.detail.annotation = await api(`/api/images/${state.current}/annotation`, 'PUT', {boxes});
    state.dirty = false;
    renderStatus();
    toast('Saved. The dataset is updated.');
    await refresh();
    if ($('jump-next').checked) {
      const items = state.library.items;
      const start = items.findIndex((item) => item.id === state.current);
      const next = [...items.slice(start + 1), ...items.slice(0, start)].find((item) => !item.annotated);
      if (next) openImage(next.id);
    }
  } catch (error) { toast(error.message, true); }
}

function step(direction) {
  const items = visibleItems();
  const index = items.findIndex((item) => item.id === state.current);
  const next = items[index + direction];
  if (next) openImage(next.id);
}

$('save-annotation').addEventListener('click', save);

$('delete-image').addEventListener('click', async () => {
  if (!state.current || !confirm('Delete this image and its boxes for everyone?')) return;
  try {
    await api(`/api/images/${state.current}`, 'DELETE');
    state.current = null; state.detail = null; state.picture = null; state.boxes = []; state.dirty = false;
    $('image-canvas').hidden = true; $('image-empty').hidden = false;
    $('image-title').textContent = 'Select an image'; $('image-meta').textContent = ''; $('thermal-panel').hidden = true;
    renderStatus(); renderBoxes();
    toast('Image deleted.');
    refresh();
  } catch (error) { toast(error.message, true); }
});

document.addEventListener('keydown', (event) => {
  if (event.target.closest('input, select, textarea, dialog') || event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === 's') { event.preventDefault(); save(); }
  else if ((key === 'delete' || key === 'backspace') && state.selected != null) { event.preventDefault(); removeBox(state.selected); }
  else if (key === 'n') step(1);
  else if (key === 'p') step(-1);
  else if (key === 'escape') { state.selected = null; renderBoxes(); draw(); }
});

window.addEventListener('beforeunload', (event) => { if (state.dirty) event.preventDefault(); });

// ---------------------------------------------------------
// TAKE A PHOTO, UPLOAD, DOWNLOAD THE DATASET
// ---------------------------------------------------------
$('take-photo').addEventListener('click', async () => {
  const button = $('take-photo');
  button.disabled = true;
  button.textContent = 'Taking photo…';
  try {
    const result = await api('/api/images/capture', 'POST');
    if (result.warning) toast(result.warning, true);
    await refresh();
    openImage(result.id);
  } catch (error) { toast(error.message, true); }
  button.textContent = 'Take photo';
  button.disabled = !state.library?.camera.available;
});

$('upload-images').addEventListener('click', () => $('upload-files').click());
$('upload-files').addEventListener('change', async () => {
  const files = [...$('upload-files').files];
  if (!files.length) return;
  const form = new FormData();
  for (const file of files) form.append('files', file);
  toast(`Uploading ${files.length} image${files.length === 1 ? '' : 's'}…`);
  try {
    const result = await api('/api/images', 'POST', form);
    await refresh();
    toast(`Uploaded ${result.ids.length}.`);
    if (!state.current && result.ids.length) openImage(result.ids[0]);
  } catch (error) { toast(error.message, true); }
  $('upload-files').value = '';
});

$('download-dataset').addEventListener('click', async () => {
  const response = await fetch('/api/images/dataset', {headers: {'X-Session': state.token || ''}});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    toast(payload.detail || 'Download failed.', true);
    if (response.status === 401) join();
    return;
  }
  const link = element('a', {href: URL.createObjectURL(await response.blob()), download: 'yolo_dataset.zip'});
  link.click();
  URL.revokeObjectURL(link.href);
});

// ---------------------------------------------------------
// START
// ---------------------------------------------------------
if (state.token) $('user-badge').textContent = state.name; else join();
refresh();
setInterval(refresh, 3000);

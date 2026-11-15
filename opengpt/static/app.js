/* openGPT front end: plain JavaScript, Chart.js for the charts. */
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const num = (v, d = 3) => (v == null || !isFinite(v) ? '–' : Number(v).toFixed(d));
const pct = v => (v == null || !isFinite(v) ? '–' : (100 * v).toFixed(1) + '%');
const big = v => (v == null ? '–' : Number(v).toLocaleString('en-US'));

async function api(path, body) {
  const r = await fetch(path, body === undefined ? undefined
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

const S = { state: null, run: null, es: null, charts: {}, loaded: {}, corpus: null, cmp: {}, dirty: false };
const COLORS = { A: '#b45309', B: '#1d4ed8', C: '#b91c1c', D: '#7c3aed', E: '#047857', F: '#0f766e' };
const PALETTE = ['#1d4ed8', '#b91c1c', '#047857', '#b45309', '#7c3aed', '#0f766e', '#be185d', '#4b5563'];

/* ---------------------------------------------------------------- tabs */
function showTab() {
  const name = (location.hash || '#learn').slice(1).split('/')[0];
  const tab = document.getElementById(name) ? name : 'learn';
  $$('.tab').forEach(t => t.classList.toggle('on', t.id === tab));
  $$('#tabs a').forEach(a => a.classList.toggle('on', a.dataset.tab === tab));
  const init = { learn: loadLesson, data: loadData, generate: fillRunSelect, runs: loadRuns, code: loadSource }[tab];
  if (init && (!S.loaded[tab] || tab === 'runs' || tab === 'generate')) { S.loaded[tab] = true; init(); }
  if (tab === 'train') Object.values(S.charts).forEach(c => c.resize && c.resize());
}
window.addEventListener('hashchange', showTab);

/* ---------------------------------------------------------------- charts */
function lineChart(canvas, datasets, yTitle, opts = {}) {
  return new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      animation: false, parsing: false, normalized: true, responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      elements: { point: { radius: 0 } },
      scales: {
        x: { type: 'linear', title: { display: true, text: 'training step' }, ticks: { maxTicksLimit: 8 } },
        y: { title: { display: !!yTitle, text: yTitle }, ...(opts.y || {}) },
      },
      plugins: { legend: { display: opts.legend !== false, labels: { boxWidth: 14, usePointStyle: false } } },
    },
  });
}
const ds = (label, color, extra = {}) => ({ label, data: [], borderColor: color, backgroundColor: color, borderWidth: 2, tension: 0, ...extra });
function thin(points, max = 1500) {
  if (points.length <= max) return points;
  const k = Math.ceil(points.length / max), out = [];
  for (let i = 0; i < points.length; i += k) out.push(points[i]);
  if (out[out.length - 1] !== points[points.length - 1]) out.push(points[points.length - 1]);
  return out;
}
function makeCharts() {
  S.charts.loss = lineChart($('#lossChart'), [
    ds('train, each step', '#bfd0f5', { borderWidth: 1 }),
    ds('train, smoothed', '#1d4ed8'),
    ds('held-out (val)', '#b91c1c', { pointRadius: 3, borderWidth: 2 }),
    ds('blind guessing', '#6b7280', { borderDash: [6, 5], borderWidth: 1.5 }),
  ], 'loss');
  S.charts.acc = lineChart($('#accChart'), [
    ds('train', '#1d4ed8', { pointRadius: 3 }),
    ds('held-out (val)', '#b91c1c', { pointRadius: 3 }),
    ds('most common word', '#6b7280', { borderDash: [6, 5], borderWidth: 1.5 }),
  ], 'share right', { y: { min: 0, ticks: { callback: v => Math.round(v * 100) + '%' } } });
  S.charts.grad = lineChart($('#gradChart'), [ds('gradient norm', '#7c3aed', { borderWidth: 1.2 })], '', { legend: false });
  S.charts.lr = lineChart($('#lrChart'), [ds('learning rate', '#047857')], '', { legend: false });
  S.charts.cmp = lineChart($('#cmpChart'), [], 'held-out loss');
}

/* ---------------------------------------------------------------- train */
const GROUPS = [
  ['Run', ['name']],
  ['Data', ['corpus', 'min_freq']],
  ['Model', ['T', 'C', 'n_layer', 'n_head', 'mlp_mult']],
  ['Training', ['steps', 'B', 'lr', 'schedule', 'warmup', 'clip', 'seed', 'dtype']],
  ['Checks', ['eval_every', 'eval_batches', 'sample_words', 'temperature']],
];
const CHOICES = { schedule: ['constant', 'cosine'], dtype: ['float32', 'float64'] };

function buildForm() {
  const { defaults, help, corpora } = S.state;
  $('#cfgForm').innerHTML = GROUPS.map(([title, keys]) => `<fieldset><legend>${title}</legend>` + keys.map(k => {
    const v = defaults[k], tip = esc(help[k] || 'A label for this run.');
    let input;
    if (k === 'corpus') input = `<select name="corpus">${corpora.map(c => `<option${c === v ? ' selected' : ''}>${esc(c)}</option>`).join('')}</select>`;
    else if (CHOICES[k]) input = `<select name="${k}">${CHOICES[k].map(c => `<option${c === v ? ' selected' : ''}>${c}</option>`).join('')}</select>`;
    else if (typeof v === 'number') input = `<input name="${k}" type="number" step="any" value="${v}">`;
    else input = `<input name="${k}" type="text" value="${esc(v)}" placeholder="optional">`;
    return `<div class="field"><label title="${tip}">${k}</label>${input}<small>${tip}</small></div>`;
  }).join('') + '</fieldset>').join('');
  $('#cfgForm').addEventListener('input', estimate);
  estimate();
}
function readCfg() {
  const cfg = {};
  $$('#cfgForm [name]').forEach(el => { cfg[el.name] = el.type === 'number' ? Number(el.value) : el.value; });
  return cfg;
}
async function estimate() {
  const c = readCfg();
  try {
    const key = c.corpus + '|' + c.min_freq;
    if (!S.vocab || S.vocab.key !== key) S.vocab = { key, V: (await api(`/api/corpus?name=${encodeURIComponent(c.corpus)}&min_freq=${c.min_freq}`)).vocab };
    const V = S.vocab.V, H = c.mlp_mult * c.C;
    const n = V * c.C + c.T * c.C + c.n_layer * (4 * c.C * c.C + 2 * c.C * H + H + 5 * c.C) + 2 * c.C + c.C * V + V;
    const bad = c.C % c.n_head ? ` · C must divide evenly by n_head` : '';
    $('#weightsEst').textContent = `Vocabulary ${big(V)} words · ${big(n)} weights${bad}`;
  } catch (e) { $('#weightsEst').textContent = e.message; }
}

function newRun(summary) {
  S.run = { id: summary.id, summary, metrics: [], evals: [], samples: [], smooth: [], ema: 0, emaN: 0 };
}
function applyDelta(d) {
  const r = S.run;
  if (!r) return;
  r.summary = d.summary;
  for (const m of d.metrics) {
    r.metrics.push(m);
    r.ema = 0.95 * r.ema + 0.05 * m[1]; r.emaN += 1;
    r.smooth.push({ x: m[0], y: r.ema / (1 - Math.pow(0.95, r.emaN)) });
  }
  r.evals.push(...d.evals);
  r.samples.push(...d.samples);
  if (!S.dirty) { S.dirty = true; requestAnimationFrame(render); }
}
function connect(id) {
  if (S.es) S.es.close();
  const es = S.es = new EventSource('/api/run/stream?id=' + encodeURIComponent(id));
  es.onmessage = e => applyDelta(JSON.parse(e.data));
  const done = async () => { es.close(); if (S.es === es) S.es = null; S.state = await api('/api/state'); render(); };
  es.addEventListener('end', done);
  es.onerror = done;
}
function render() {
  S.dirty = false;
  const r = S.run, pill = $('#pill');
  if (!r) { pill.className = 'pill idle'; pill.textContent = 'no run'; return; }
  const s = r.summary, info = s.info || {}, last = r.evals[r.evals.length - 1] || {};
  const alive = ['starting', 'running', 'paused'].includes(s.status);
  pill.className = 'pill ' + s.status;
  pill.textContent = `${s.status} · step ${big(s.step)} / ${big(s.steps)}`;
  const eta = alive && s.sps > 0 ? Math.round((s.steps - s.step) / s.sps) : null;
  const card = (k, v) => `<div class="card"><small>${k}</small><b>${v}</b></div>`;
  $('#status').innerHTML = [
    card('status', esc(s.status)), card('step', `${big(s.step)} / ${big(s.steps)}`),
    card('steps / sec', num(s.sps, 1)), card('time left', eta == null ? '–' : eta > 90 ? Math.round(eta / 60) + ' min' : eta + ' s'),
    card('train loss', num(r.smooth.length ? r.smooth[r.smooth.length - 1].y : null)),
    card('val loss', num(last.val_loss)), card('val accuracy', pct(last.val_acc)),
    card('weights', big(info.n_params)), card('vocabulary', big(info.V)),
  ].join('');
  $('#bar').style.width = (100 * s.step / Math.max(1, s.steps)) + '%';
  $('#trainMsg').textContent = s.error || '';
  $('#btnStart').disabled = alive;
  $('#btnPause').disabled = !alive || s.status === 'starting';
  $('#btnPause').textContent = s.status === 'paused' ? 'Resume' : 'Pause';
  $('#btnStop').disabled = !alive;

  const C = S.charts, span = Math.max(s.steps, s.step, 1);
  C.loss.data.datasets[0].data = thin(r.metrics.map(m => ({ x: m[0], y: m[1] })));
  C.loss.data.datasets[1].data = thin(r.smooth);
  C.loss.data.datasets[2].data = r.evals.map(e => ({ x: e.step, y: e.val_loss }));
  C.loss.data.datasets[3].data = info.chance_loss ? [{ x: 0, y: info.chance_loss }, { x: span, y: info.chance_loss }] : [];
  C.acc.data.datasets[0].data = r.evals.map(e => ({ x: e.step, y: e.train_acc }));
  C.acc.data.datasets[1].data = r.evals.map(e => ({ x: e.step, y: e.val_acc }));
  C.acc.data.datasets[2].data = info.baseline_acc != null ? [{ x: 0, y: info.baseline_acc }, { x: span, y: info.baseline_acc }] : [];
  C.grad.data.datasets[0].data = thin(r.metrics.map(m => ({ x: m[0], y: m[2] })));
  C.lr.data.datasets[0].data = thin(r.metrics.map(m => ({ x: m[0], y: m[3] })));
  for (const k of ['loss', 'acc', 'grad', 'lr']) { C[k].options.scales.x.max = span; C[k].update('none'); }

  if (r.evals.length) {
    const chk = (ok, text) => `<div class="check"><b class="${ok == null ? 'wait' : ok ? 'ok' : 'no'}">${ok == null ? 'wait' : ok ? 'yes' : 'no'}</b><span>${text}</span></div>`;
    const trained = last.step > 0, gc = S.gradcheck;
    $('#checks').innerHTML =
      chk(gc ? gc.ok : null, gc ? `Gradients match measured slopes (largest gap ${gc.error.toExponential(1)}).` : 'Gradients match measured slopes: run the check in the Code tab.') +
      chk(trained ? last.val_loss < info.chance_loss : null, `Held-out loss ${num(last.val_loss)} is below blind guessing, ln(V) = ${num(info.chance_loss)}.`) +
      chk(trained ? last.val_acc > info.baseline_acc : null, `Held-out accuracy ${pct(last.val_acc)} beats always saying "${esc(info.baseline_word)}" (${pct(info.baseline_acc)}).`) +
      chk(trained ? Math.abs(last.train_acc - last.val_acc) < 0.08 : null, `Train ${pct(last.train_acc)} and held-out ${pct(last.val_acc)} accuracy are within 8 points: patterns, not memory.`);
  }
  if (r.samples.length) {
    $('#samples').innerHTML = r.samples.slice(-4).reverse().map(x => `<div class="sample"><small>step ${big(x.step)}</small>${esc(x.text)}</div>`).join('');
  }
}
async function startRun() {
  $('#trainMsg').textContent = '';
  try {
    const summary = await api('/api/run/start', { cfg: readCfg() });
    newRun(summary); $('#checks').textContent = 'Waiting for the first check…'; $('#samples').textContent = '';
    render(); connect(summary.id);
  } catch (e) { $('#trainMsg').textContent = e.message; }
}
async function control(what) {
  try {
    const s = await api('/api/run/' + what, { id: S.run.id });
    S.run.summary = s; render();
  } catch (e) { $('#trainMsg').textContent = e.message; }
}

/* ---------------------------------------------------------------- learn */
async function loadLesson() {
  $('#lesson').innerHTML = '<p class="muted">Running the example…</p>';
  let L;
  try { L = await api('/api/lesson'); } catch (e) { $('#lesson').innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  const ex = L.example;
  const exVal = { B: ex.B, T: ex.T, C: ex.C, V: ex.V, H: ex.H, N: ex.B * ex.T };
  $('#glossary').innerHTML = '<div class="box"><h3>The six letters used in every shape</h3><table class="grid"><tr><th>Letter</th><th>Name</th><th>What it means</th><th>In the example</th></tr>' +
    L.glossary.map(([s, n, d]) => `<tr><td class="sym">${s}</td><td><b>${esc(n)}</b></td><td>${esc(d)}</td><td class="sym">${exVal[s]}</td></tr>`).join('') +
    '</table><p class="muted">A shape such as (B,T,C) reads: B windows, each with T positions, each position holding C numbers. Every other word used here is explained just below.</p></div>' +
    '<div class="box"><h3>Every word this lesson uses</h3><table class="grid"><tr><th>Word</th><th>What it means</th></tr>' +
    L.terms.map(([w, d]) => `<tr><td style="white-space:nowrap"><b>${esc(w)}</b></td><td>${esc(d)}</td></tr>`).join('') + '</table></div>';
  const counts = {};
  L.steps.forEach(s => { (counts[s.stage] = counts[s.stage] || []).push(s.n); });
  const range = a => (a.length === 1 ? `step ${a[0]}` : `steps ${a[0]} to ${a[a.length - 1]}`);
  $('#stageNav').innerHTML = Object.entries(L.stages).map(([k, [name]]) =>
    `<a href="#learn/${k}" data-stage="${k}" style="--c:${COLORS[k]}"><b>${k}</b> ${esc(name)}<span>${range(counts[k]).replace(/steps? /, '')}</span></a>`).join('');
  let html = '', last = '';
  for (const s of L.steps) {
    if (s.stage !== last) {
      const [name, sub] = L.stages[s.stage];
      html += `<div class="stage" id="stage${s.stage}" style="--c:${COLORS[s.stage]}"><span class="letter">${s.stage}</span><div><h2>${esc(name)}</h2><p>${esc(sub)} · ${range(counts[s.stage])}</p></div></div>`;
      last = s.stage;
    }
    const note = s.note ? `<p class="hand"><b>${s.check ? 'Check it by hand:' : 'Note:'}</b> ${esc(s.note)}</p>` : '';
    const mini = s.curve ? `<div class="mini"><canvas id="curve${s.n}"></canvas></div>` : '';
    html += `<section class="step" id="s${s.n}" style="--c:${COLORS[s.stage]}">
      <header><span class="num">${s.n}</span><h3>${esc(s.title)}</h3><span class="shape">${esc(s.shape)}</span></header>
      <p class="why">${s.why}</p><div class="cols">
      <div class="panel math"><h4>Math</h4><div class="formula">${s.math}</div></div>
      <div class="panel code"><h4>Python</h4><pre>${s.code}</pre></div>
      <div class="panel ex"><h4>Example: ${esc(L.text)}</h4><pre>${esc(s.example)}</pre>${mini}${note}</div>
      </div></section>`;
  }
  $('#lesson').innerHTML = html;
  for (const s of L.steps.filter(x => x.curve)) {
    const c = lineChart($('#curve' + s.n), [
      ds('loss', '#1d4ed8', { data: s.curve.map((y, x) => ({ x, y })) }),
      ds(`guessing`, '#6b7280', { borderDash: [6, 5], borderWidth: 1.5, data: [{ x: 0, y: s.chance }, { x: s.curve.length - 1, y: s.chance }] }),
    ], '', { y: { min: 0 } });
    c.options.scales.x.title.text = 'step'; c.update('none');
  }
  $$('#stageNav a').forEach(a => a.addEventListener('click', e => {
    e.preventDefault(); document.getElementById('stage' + a.dataset.stage).scrollIntoView();
  }));
}
function colToggles() {
  $$('#cols input').forEach(cb => cb.addEventListener('change', () =>
    $('#lesson').classList.toggle('hide-' + cb.dataset.col, !cb.checked)));
  $('#reloadLesson').addEventListener('click', loadLesson);
}

/* ---------------------------------------------------------------- data */
async function loadData() {
  $('#dCorpus').innerHTML = S.state.corpora.map(c => `<option>${esc(c)}</option>`).join('');
  $('#dFetch').hidden = S.state.corpora.includes('tinystories');
  await showCorpus(); tokenize();
}
async function showCorpus() {
  $('#dMsg').textContent = 'Reading…';
  try {
    const c = S.corpus = await api(`/api/corpus?name=${encodeURIComponent($('#dCorpus').value)}&min_freq=${$('#dMinFreq').value}`);
    const card = (k, v) => `<div class="card"><small>${k}</small><b>${v}</b></div>`;
    $('#dStats').innerHTML = [
      card('words in the text', big(c.tokens)), card('distinct words', big(c.distinct_words)),
      card('vocabulary V', big(c.vocab)), card('turned into &lt;unk&gt;', pct(c.unk_rate)),
      card('train (90%)', big(c.train_tokens)), card('held out (10%)', big(c.val_tokens)),
      card('blind guessing loss', num(c.chance_loss, 2)),
      card(`always "${esc(c.baseline_word)}"`, pct(c.baseline_acc)),
    ].join('');
    const max = c.top[0][1];
    $('#dTop').innerHTML = c.top.slice(0, 20).map(([w, n]) => `<div class="bar"><span class="mono">${esc(w)}</span><i style="width:${100 * n / max}%"></i><span>${big(n)}</span></div>`).join('');
    $('#dSample').textContent = c.sample;
    $('#dMsg').textContent = '';
  } catch (e) { $('#dMsg').textContent = e.message; }
}
async function tokenize() {
  try {
    const t = await api('/api/tokenize', { text: $('#tokText').value, corpus: $('#dCorpus').value, min_freq: Number($('#dMinFreq').value), T: Number($('#tokT').value) });
    let h = `<p class="muted">${t.tokens.length} tokens${t.unknown.length ? `, ${t.unknown.length} not in the vocabulary (shown in red, id 0)` : ''}</p><div class="chips">` +
      t.tokens.map((w, i) => `<span class="chip${t.ids[i] === 0 ? ' unk' : ''}">${esc(w)}<small>${t.ids[i]}</small></span>`).join('') + '</div>';
    if (t.window) h += `<p class="muted">The first window of T = ${t.T}: the model reads x and must answer y, the same words moved one place on.</p>
      <div class="chips"><b>x</b>${t.window.x.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</div>
      <div class="chips"><b>y</b>${t.window.y.map(w => `<span class="chip y">${esc(w)}</span>`).join('')}</div>`;
    $('#tokOut').innerHTML = h;
  } catch (e) { $('#tokOut').innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
async function fetchStories() {
  $('#dMsg').textContent = 'Downloading TinyStories, about 19 MB…';
  try {
    const r = await api('/api/fetch', {});
    S.state.corpora = r.corpora; S.vocab = null;
    await loadData(); buildForm();
    $('#dMsg').textContent = 'Downloaded. Pick tinystories; min_freq 10 is a good start.';
  } catch (e) { $('#dMsg').textContent = e.message; }
}

/* ---------------------------------------------------------------- generate */
async function fillRunSelect() {
  const runs = (await api('/api/runs')).runs.filter(r => r.step > 0 || r.status !== 'error');
  const keep = $('#gRun').value;
  $('#gRun').innerHTML = runs.length
    ? runs.map(r => `<option value="${esc(r.id)}">${esc(r.id)}${r.name ? ' · ' + esc(r.name) : ''} · ${r.status} · step ${big(r.step)}</option>`).join('')
    : '<option value="">no runs yet: train one first</option>';
  if (S.pick || keep) $('#gRun').value = S.pick || keep;
  S.pick = null;
}
async function generate() {
  $('#gMsg').textContent = '';
  try {
    const g = await api('/api/generate', { id: $('#gRun').value, prompt: $('#gPrompt').value, n: Number($('#gN').value), temperature: Number($('#gTemp').value), seed: Number($('#gSeed').value) });
    const start = g.prompt_tokens.map(w => `<span class="w prompt">${esc(w)}</span>`).join(' ');
    const words = g.steps.map(s => {
      const tip = `chosen with p = ${s.p.toFixed(2)}\n` + s.top.map(([w, p]) => `${w}  ${p.toFixed(2)}`).join('\n');
      return `<span class="w" title="${esc(tip)}" style="background:rgba(29,78,216,${(0.05 + 0.3 * (1 - s.p)).toFixed(2)})">${esc(s.drawn)}</span>`;
    }).join(' ');
    $('#gText').classList.remove('muted');
    $('#gText').innerHTML = start + ' ' + words +
      `<p class="muted" style="font-size:13px">Darker = the model was less sure. Hover a word for its odds.${g.unknown.length ? ' Not in the vocabulary: ' + esc(g.unknown.join(', ')) + '.' : ''}</p>`;
    const max = g.next[0][1];
    $('#gNext').innerHTML = g.next.map(([w, p]) => `<div class="bar"><span class="mono">${esc(w)}</span><i style="width:${100 * p / max}%"></i><span>${pct(p)}</span></div>`).join('');
    drawAttention(g);
  } catch (e) { $('#gMsg').textContent = e.message; }
}
function drawAttention(g) {
  const T = g.context.length, cell = Math.max(5, Math.min(18, Math.floor(200 / T)));
  let html = `<p class="muted">Context, top to bottom and left to right: <span class="mono">${g.context.map(esc).join(' · ')}</span></p><div class="att">`;
  g.attention.forEach((heads, l) => heads.forEach((_, h) => {
    html += `<figure><figcaption>block ${l + 1} · head ${h + 1}</figcaption><canvas data-l="${l}" data-h="${h}" width="${T * cell}" height="${T * cell}"></canvas></figure>`;
  }));
  $('#gAtt').innerHTML = html + '</div>';
  $$('#gAtt canvas').forEach(cv => {
    const m = g.attention[cv.dataset.l][cv.dataset.h], ctx = cv.getContext('2d');
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
    for (let t = 0; t < T; t++) for (let s = 0; s < T; s++) {
      ctx.fillStyle = s > t ? '#f1efe9' : `rgba(29,78,216,${m[t][s]})`;
      ctx.fillRect(s * cell, t * cell, cell - 1, cell - 1);
    }
    cv.title = 'Bottom row: what the last word looks at when predicting the next one.';
  });
}

/* ---------------------------------------------------------------- runs */
async function loadRuns() {
  const runs = (await api('/api/runs')).runs;
  const head = ['', 'run', 'name', 'status', 'steps', 'corpus', 'blocks', 'heads', 'C', 'T', 'lr', 'weights', 'val loss', 'val acc', ''];
  $('#runsTable').innerHTML = '<tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr>' + (runs.length ? runs.map(r => {
    const c = r.cfg, l = r.last || {};
    return `<tr><td><input type="checkbox" data-id="${esc(r.id)}"${S.cmp[r.id] ? ' checked' : ''}></td>
      <td class="mono">${esc(r.id)}</td><td>${esc(r.name || '')}</td><td>${esc(r.status)}${r.error ? ': ' + esc(r.error) : ''}</td>
      <td>${big(r.step)} / ${big(r.steps)}</td><td>${esc(c.corpus)}</td><td>${c.n_layer}</td><td>${c.n_head}</td><td>${c.C}</td><td>${c.T}</td>
      <td>${c.lr}</td><td>${big((r.info || {}).n_params)}</td><td>${num(l.val_loss)}</td><td>${pct(l.val_acc)}</td>
      <td><button class="ghost" data-gen="${esc(r.id)}">Generate</button></td></tr>`;
  }).join('') : '<tr><td colspan="15" class="muted">No runs yet. Start one in the Train tab.</td></tr>');
  $$('#runsTable input').forEach(cb => cb.addEventListener('change', () => toggleCompare(cb.dataset.id, cb.checked)));
  $$('#runsTable [data-gen]').forEach(b => b.addEventListener('click', () => { S.pick = b.dataset.gen; location.hash = '#generate'; }));
  if (!Object.keys(S.cmp).length) for (const r of runs.slice(0, 3)) await toggleCompare(r.id, true, true);
  $$('#runsTable input').forEach(cb => { cb.checked = !!S.cmp[cb.dataset.id]; });
  drawCompare();
}
async function toggleCompare(id, on, quiet) {
  if (on) { try { S.cmp[id] = await api('/api/run?id=' + encodeURIComponent(id)); } catch (e) { return; } }
  else delete S.cmp[id];
  if (!quiet) drawCompare();
}
function drawCompare() {
  S.charts.cmp.data.datasets = Object.entries(S.cmp).map(([id, rec], i) => {
    const c = rec.summary.cfg;
    return ds(`${rec.summary.name || id} · ${c.n_layer} blocks, ${c.n_head} heads, C ${c.C}`, PALETTE[i % PALETTE.length],
      { pointRadius: 3, data: rec.evals.map(e => ({ x: e.step, y: e.val_loss })) });
  });
  S.charts.cmp.update('none');
}

/* ---------------------------------------------------------------- code */
async function loadSource() {
  const { files } = await api('/api/source');
  const show = i => {
    $$('#fileList a').forEach((a, j) => a.classList.toggle('on', i === j));
    $('#fileView').innerHTML = files[i].html;
    $('#fileView').scrollTop = 0;
  };
  $('#fileList').innerHTML = files.map(f => `<a href="#code"><small>${f.lines}</small>${esc(f.path.replace('opengpt/', ''))}</a>`).join('');
  $$('#fileList a').forEach((a, i) => a.addEventListener('click', e => { e.preventDefault(); show(i); }));
  show(0);
}
async function gradcheck() {
  $('#gcMsg').textContent = 'Checking…'; $('#gcMsg').className = 'muted';
  try {
    const r = S.gradcheck = await api('/api/gradcheck', { n_layer: Number($('#gcL').value), n_head: Number($('#gcH').value) });
    $('#gcMsg').className = r.ok ? '' : 'err';
    $('#gcMsg').textContent = `${r.ok ? 'Correct' : 'WRONG'}: largest gap ${r.error.toExponential(1)} over ${big(r.weights)} weights (${r.n_layer} blocks, ${r.n_head} heads).`;
    if (S.run) render();
  } catch (e) { $('#gcMsg').className = 'err'; $('#gcMsg').textContent = e.message; }
}

/* ---------------------------------------------------------------- start */
async function main() {
  makeCharts(); colToggles();
  $('#btnStart').addEventListener('click', startRun);
  $('#btnPause').addEventListener('click', () => control(S.run.summary.status === 'paused' ? 'resume' : 'pause'));
  $('#btnStop').addEventListener('click', () => control('stop'));
  $('#dLoad').addEventListener('click', async () => { await showCorpus(); tokenize(); });
  $('#dCorpus').addEventListener('change', async () => { await showCorpus(); tokenize(); });
  $('#dFetch').addEventListener('click', fetchStories);
  $('#tokGo').addEventListener('click', tokenize);
  $('#gGo').addEventListener('click', generate);
  $('#gcGo').addEventListener('click', gradcheck);
  S.state = await api('/api/state');
  buildForm();
  if (S.state.active) {
    newRun(S.state.active);
    if (['starting', 'running', 'paused'].includes(S.state.active.status)) connect(S.state.active.id);
    else applyDelta(await api('/api/run?id=' + encodeURIComponent(S.state.active.id)));
  }
  render();
  showTab();
}
main().catch(e => { document.body.insertAdjacentHTML('afterbegin', `<p class="err" style="padding:20px">openGPT could not start: ${esc(e.message)}</p>`); });

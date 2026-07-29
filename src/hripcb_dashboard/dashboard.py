"""A dependency-free, generic parameter-sweep dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _keys(records: list[dict], section: str) -> list[str]:
    keys: set[str] = set()
    for record in records:
        keys.update((record.get(section) or {}).keys())
    return sorted(keys)


def _safe_json(records: list[dict]) -> str:
    return json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")


def write_dashboard_html(
    output_dir: Path,
    records: list[dict],
    *,
    title: str = "Preprocessing Parameter Sweep",
    primary_metric: str = "map50_95",
) -> Path:
    """Write a generic dashboard that renders arbitrary modules and parameters."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_keys = _keys(records, "parameters")
    metric_keys = _keys(records, "metrics")
    payload = _safe_json(records)
    escaped_title = html.escape(title)
    metric_labels = {
        "map50_95": "mAP50-95",
        "map50": "mAP50",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    primary_metric_label = metric_labels.get(primary_metric, primary_metric)
    parameter_json = json.dumps(parameter_keys)
    metric_json = json.dumps(metric_keys)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{ color-scheme: light; --canvas: #f4f7fb; --ink: #172033; --muted: #697586; --panel: rgba(255,255,255,.82); --panel-solid: #ffffff; --line: #dce5ef; --blue: #2563eb; --green: #198754; --amber: #a86300; --shadow: 0 18px 45px rgba(40, 64, 92, .10); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; background: radial-gradient(circle at 8% 0%, rgba(207, 229, 255, .72), transparent 34%), radial-gradient(circle at 92% 10%, rgba(215, 245, 232, .62), transparent 32%), var(--canvas); color: var(--ink); }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 38px 28px 70px; }}
    .eyebrow {{ color: var(--blue); font: 700 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .14em; text-transform: uppercase; }}
    h1 {{ margin: 10px 0 8px; max-width: 850px; font-size: clamp(30px, 4vw, 52px); line-height: .98; letter-spacing: -.045em; }}
    .intro {{ max-width: 780px; color: var(--muted); line-height: 1.6; }}
    .hero {{ display: grid; grid-template-columns: 1.35fr .65fr; gap: 16px; margin: 28px 0 18px; }}
    .hero-card, .panel {{ border: 1px solid rgba(255,255,255,.92); background: var(--panel); backdrop-filter: blur(20px) saturate(130%); -webkit-backdrop-filter: blur(20px) saturate(130%); border-radius: 20px; box-shadow: var(--shadow); }}
    .hero-card {{ min-height: 178px; padding: 24px; }} .hero-card:first-child {{ background: linear-gradient(135deg, rgba(255,255,255,.94), rgba(236,247,255,.86)); }} .hero-card h2 {{ margin: 8px 0 4px; font-size: 25px; }} .hero-card p {{ margin: 0; color: var(--muted); }}
    .winner-number {{ margin-top: 20px; color: var(--blue); font: 700 42px/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: -.05em; }} .winner-label {{ margin-top: 7px; color: var(--muted); font-size: 12px; }}
    .controls {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)) 1.4fr 1fr 1.6fr; gap: 10px; margin: 18px 0; }}
    .active-card {{ display: grid; grid-template-columns: 1.4fr repeat(5, 1fr); gap: 10px; margin: 18px 0; }} .active-card > div {{ padding: 15px; border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,.78); }} .active-card .active-main {{ background: linear-gradient(135deg, #ffffff, #eaf2ff); }} .active-card strong {{ display: block; margin-top: 6px; font-size: 16px; }} .protocol {{ color: var(--green); font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace; }}
    label {{ display: grid; gap: 6px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    select, input {{ width: 100%; border: 1px solid var(--line); border-radius: 12px; padding: 11px 12px; color: var(--ink); background: rgba(255,255,255,.88); font: inherit; outline: none; box-shadow: 0 4px 12px rgba(40, 64, 92, .04); }}
    select:focus, input:focus {{ border-color: var(--blue); box-shadow: 0 0 0 4px rgba(37,99,235,.13); }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(270px, .6fr); gap: 16px; align-items: start; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.86); box-shadow: var(--shadow); }} table {{ width: 100%; border-collapse: collapse; min-width: 880px; }}
    th, td {{ padding: 13px; border-bottom: 1px solid rgba(220,229,239,.78); text-align: left; white-space: nowrap; }}
    th {{ position: sticky; top: 0; z-index: 1; color: var(--muted); background: rgba(248,250,253,.96); font: 700 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .08em; text-transform: uppercase; }}
    tbody tr {{ cursor: pointer; transition: background .15s ease, transform .15s ease; }} tbody tr:hover {{ background: #f4f8ff; }} tbody tr.selected {{ background: #eaf2ff; }}
    tbody tr.best td:first-child::before {{ content: "BEST"; margin-right: 7px; color: #ffffff; background: var(--green); border-radius: 5px; padding: 3px 5px; font: 800 9px ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .metric {{ color: var(--green); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .detail {{ position: sticky; top: 18px; padding: 18px; }} .detail h2 {{ margin: 0 0 4px; font-size: 20px; }} .detail .sub {{ color: var(--muted); font-size: 12px; }}
    .preview {{ width: 100%; margin: 16px 0; aspect-ratio: 4 / 3; border: 1px solid var(--line); border-radius: 14px; object-fit: contain; background: #f8fafc; }}
    .kv {{ display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; margin-top: 14px; font-size: 12px; }} .kv dt {{ color: var(--muted); }} .kv dd {{ margin: 0; color: var(--ink); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }}
    .empty {{ padding: 28px; color: var(--muted); text-align: center; }} .footer {{ margin-top: 22px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 1100px) {{ .active-card {{ grid-template-columns: repeat(3, 1fr); }} }} @media (max-width: 900px) {{ .hero, .layout {{ grid-template-columns: 1fr; }} .detail {{ position: static; }} .active-card {{ grid-template-columns: repeat(2, 1fr); }}}} @media (max-width: 650px) {{ main {{ padding: 24px 14px 50px; }} .controls, .active-card {{ grid-template-columns: 1fr; }}}}
    @media (prefers-reduced-transparency: reduce) {{ .hero-card, .panel, .table-wrap {{ background: var(--panel-solid); backdrop-filter: none; -webkit-backdrop-filter: none; }}}}
    @media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; transition: none !important; }}}}
  </style>
</head>
<body>
<main>
  <div class="eyebrow">Image Processing Lab / Parameter Search</div>
  <h1>{escaped_title}</h1>
  <p class="intro">A shared comparison surface for every preprocessing module. The primary ranking and Best run use combined techniques only; original, noise-only and contrast-only records remain available as reference runs. The primary metric is <strong>{html.escape(primary_metric_label)}</strong>.</p>
  <section class="hero">
    <article class="hero-card"><div class="eyebrow">Current best run</div><h2 id="winnerName">—</h2><p id="winnerTechnique">Waiting for results</p><div class="winner-number" id="winnerScore">—</div><div class="winner-label">{html.escape(primary_metric_label)}</div></article>
    <article class="hero-card"><div class="eyebrow">Sweep coverage</div><h2 id="runCount">0</h2><p id="coverageText">parameter combinations</p><div class="winner-number" id="moduleCount">0</div><div class="winner-label">member modules represented · baseline control separate</div></article>
  </section>
  <section class="active-card" aria-label="Active experiment">
    <div class="active-main"><div class="eyebrow">Active experiment</div><strong id="activeName">—</strong><p class="sub" id="activeTechnique">Choose a row to inspect it.</p></div>
    <div><div class="eyebrow">Image size</div><strong>1024</strong><div class="protocol">imgsz 1024</div></div>
    <div><div class="eyebrow">Confidence</div><strong>0.25</strong><div class="protocol">conf 0.25</div></div>
    <div><div class="eyebrow">IoU</div><strong>0.70</strong><div class="protocol">IoU 0.70</div></div>
    <div><div class="eyebrow">Split</div><strong id="activeSplit">—</strong><div class="protocol">dataset</div></div>
    <div><div class="eyebrow">Primary score</div><strong id="activeScore">—</strong><div class="protocol">mAP50-95</div></div>
  </section>
  <section class="controls" aria-label="Dashboard filters">
    <label>Model<select id="modelFilter"><option value="all">All models</option></select></label>
    <label>Split<select id="splitFilter"><option value="all">All splits</option></select></label>
    <label>Module<select id="moduleFilter"><option value="all">All modules</option></select></label>
    <label>Technique<select id="techniqueFilter"><option value="all">All techniques</option></select></label>
    <label>Run type<select id="runTypeFilter"><option value="all">All runs</option><option value="combined">Combined + original baseline</option><option value="reference">Reference runs</option></select></label>
    <label>Sort<input id="sortMetric" list="metricOptions" value="{html.escape(primary_metric)}" aria-label="Sort metric"><datalist id="metricOptions"></datalist></label>
    <label>Order<select id="sortDirection"><option value="desc">High to low</option><option value="asc">Low to high</option></select></label>
    <label>Search parameters or run<input id="searchFilter" type="search" placeholder="e.g. sigma, bbhe, bilateral"></label>
  </section>
  <section class="layout">
    <div class="table-wrap"><table><thead id="tableHead"></thead><tbody id="tableBody"></tbody></table><div class="empty" id="emptyState" hidden>No matching parameter runs.</div></div>
    <aside class="panel detail"><div class="eyebrow">Selected run</div><h2 id="detailName">—</h2><div class="sub" id="detailTechnique">Choose a row to inspect it.</div><img class="preview" id="detailPreview" alt="Selected preprocessing preview"><dl class="kv" id="detailValues"></dl></aside>
  </section>
  <p class="footer">Generated as a static HTML file. No server, Node.js, or external library is required.</p>
</main>
<script>
const records = {payload};
const parameterKeys = {parameter_json};
const metricKeys = {metric_json};
const primaryMetric = {json.dumps(primary_metric)};
const labels = {{map50_95: 'mAP50-95', map50: 'mAP50', precision: 'Precision', recall: 'Recall', f1: 'F1', milliseconds: 'Time (ms)', mean_psnr: 'Mean PSNR', mean_ssim: 'Mean SSIM', model_id: 'Model', split: 'Split'}};
const state = {{model: 'all', split: 'all', module: 'all', technique: 'all', runType: 'all', search: '', sortMetric: primaryMetric, direction: 'desc', selected: null}};
const filterFields = ['model', 'split', 'module', 'technique'];
const combinedTechniques = new Set(['gaussian_bbhe', 'median_clahe', 'bilateral_agcwd', 'nlm_msr']);
const runType = record => record.is_combined === true || record.evaluation_stage === 'combined' || combinedTechniques.has(String(record.technique || '').toLowerCase()) ? 'combined' : 'reference';
const isSharedOriginal = record => record.model_id === 'baseline' && record.split === 'val' && String(record.technique || '').toLowerCase() === 'original' && ['member1', 'member2', 'member3', 'member4'].includes(record.module);
const firstSharedOriginal = records.find(record => isSharedOriginal(record));
const keepDisplayRecord = record => !(state.module === 'all' && isSharedOriginal(record) && record !== firstSharedOriginal);
const displayModule = record => isSharedOriginal(record) && state.module === 'all' ? 'baseline control' : (record.module || 'unknown');
const displayTechnique = record => isSharedOriginal(record) && state.module === 'all' ? 'Original / Shared baseline control' : (record.technique || 'unknown');
const matchesRunType = record => (state.runType === 'combined' && isSharedOriginal(record)) || state.runType === 'all' || runType(record) === state.runType;
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[char]));
const label = key => labels[key] || key.replaceAll('_', ' ').replace(/\\b\\w/g, char => char.toUpperCase());
const fmt = value => typeof value === 'number' ? value.toFixed(4) : (value ?? '—');
const modelValue = record => record.model_id || 'baseline';
const splitValue = record => record.split || 'unknown';
const metric = record => Number(record.metrics?.[state.sortMetric] ?? -Infinity);
const sortedRecords = () => [...records].sort((a,b) => state.direction === 'asc' ? metric(a) - metric(b) : metric(b) - metric(a));
const fieldValue = (record, field) => field === 'model' ? modelValue(record) : field === 'split' ? splitValue(record) : String(record[field] || 'unknown');
const recordsForFields = fields => records.filter(record => keepDisplayRecord(record) && matchesRunType(record) && fields.every(field => state[field] === 'all' || fieldValue(record, field) === state[field]));
function optionValuesFor(field) {{ const index = filterFields.indexOf(field); return [...new Set(recordsForFields(filterFields.slice(0, index)).map(record => fieldValue(record, field)))].filter(Boolean).sort(); }}
function normalizeSelection() {{ for (const field of filterFields) {{ const values = optionValuesFor(field); if (state[field] !== 'all' && !values.includes(state[field])) state[field] = 'all'; }} }}
function filterOptionLabel(field) {{ return field === 'model' ? 'All models' : field === 'split' ? 'All splits' : field === 'module' ? 'All modules' : 'All techniques'; }}
function syncCascadedFilters() {{ normalizeSelection(); for (const field of filterFields) {{ const select = document.getElementById(field + 'Filter'); const previous = state[field]; const values = optionValuesFor(field); select.innerHTML = '<option value=\"all\">' + filterOptionLabel(field) + '</option>' + values.map(value => '<option value=\"' + esc(value) + '\">' + esc(value) + '</option>').join(''); state[field] = values.includes(previous) ? previous : 'all'; select.value = state[field]; }} normalizeSelection(); }}
const matches = record => {{ const text = JSON.stringify(record).toLowerCase(); return keepDisplayRecord(record) && matchesRunType(record) && filterFields.every(field => state[field] === 'all' || fieldValue(record, field) === state[field]) && (!state.search || text.includes(state.search)); }};
function unique(key) {{ return [...new Set(records.map(record => key === 'model_id' ? modelValue(record) : key === 'split' ? splitValue(record) : record[key]))].filter(Boolean).sort(); }}
function fillSelect(id, values) {{ const select = document.getElementById(id); for (const value of values) {{ const option = document.createElement('option'); option.value = value; option.textContent = value; select.appendChild(option); }} }}
function renderHead() {{ const columns = ['id', 'model_id', 'module', 'technique', 'split', ...parameterKeys, ...metricKeys]; document.getElementById('tableHead').innerHTML = '<tr>' + columns.map(key => `<th>${{esc(label(key))}}</th>`).join('') + '</tr>'; }}
function renderActive(record) {{ const name = document.getElementById('activeName'); const technique = document.getElementById('activeTechnique'); const split = document.getElementById('activeSplit'); const score = document.getElementById('activeScore'); if (!record) {{ name.textContent = 'No matching experiment'; technique.textContent = 'Reset filters or choose another combination.'; split.textContent = '—'; score.textContent = '—'; return; }} name.textContent = record.id; technique.textContent = (record.model_label || modelValue(record)) + ' / ' + (record.module || 'module') + ' / ' + (record.technique || 'technique'); split.textContent = splitValue(record); score.textContent = fmt(record.metrics?.[primaryMetric]); }}
function renderDetail(record) {{ if (!record) {{ state.selected = null; renderActive(null); document.getElementById('detailName').textContent = '—'; document.getElementById('detailTechnique').textContent = 'No experiment selected.'; document.getElementById('detailPreview').removeAttribute('src'); document.getElementById('detailValues').innerHTML = ''; return; }} state.selected = record.id; renderActive(record); document.getElementById('detailName').textContent = record.id; document.getElementById('detailTechnique').textContent = (record.model_label || modelValue(record)) + ' / ' + displayModule(record) + ' / ' + displayTechnique(record); const image = document.getElementById('detailPreview'); image.src = record.preview || ''; image.alt = record.id + ' preview'; const meta = [['Model', record.model_label || modelValue(record)], ['Split', splitValue(record)], ['Evaluation type', record.evaluation_type || '—']]; const values = [...meta, ...parameterKeys.map(key => [label(key), record.parameters?.[key]]), ...metricKeys.map(key => [label(key), fmt(record.metrics?.[key])])]; document.getElementById('detailValues').innerHTML = values.map(([key,value]) => '<dt>' + esc(key) + '</dt><dd>' + esc(value) + '</dd>').join(''); document.querySelectorAll('tbody tr').forEach(row => row.classList.toggle('selected', row.dataset.id === record.id)); }}
function valueFor(record, key) {{ if (key === 'id') return record.id; if (key === 'model_id') return modelValue(record); if (key === 'module') return displayModule(record); if (key === 'technique') return displayTechnique(record); if (key === 'split') return splitValue(record); return record.parameters?.[key] ?? record.metrics?.[key] ?? '—'; }}
function renderRows() {{ const rows = sortedRecords().filter(matches); const bestId = rows.find(record => runType(record) === 'combined')?.id || rows[0]?.id; const columns = ['id', 'model_id', 'module', 'technique', 'split', ...parameterKeys, ...metricKeys]; const body = document.getElementById('tableBody'); body.innerHTML = rows.map(record => '<tr data-id="' + esc(record.id) + '" class="' + (record.id === bestId ? 'best' : '') + '">' + columns.map(key => '<td class="' + (metricKeys.includes(key) ? 'metric' : '') + '">' + esc(valueFor(record, key)) + '</td>').join('') + '</tr>').join(''); document.getElementById('emptyState').hidden = rows.length !== 0; body.querySelectorAll('tr').forEach(row => row.addEventListener('click', () => renderDetail(records.find(record => record.id === row.dataset.id)))); if (!state.selected || !rows.some(row => row.id === state.selected)) renderDetail(rows[0] || null); }}
function renderSummary() {{ const ranked = sortedRecords().filter(matches); const combinedRanked = ranked.filter(record => runType(record) === 'combined'); const best = combinedRanked[0]; const memberModules = new Set(ranked.map(record => record.module).filter(module => module && module !== 'baseline')); document.getElementById('runCount').textContent = ranked.length; document.getElementById('moduleCount').textContent = memberModules.size; document.getElementById('coverageText').textContent = parameterKeys.length + ' parameter dimensions / ' + metricKeys.length + ' metrics · table includes reference runs · Best run uses combined only'; if (best) {{ document.getElementById('winnerName').textContent = best.id; document.getElementById('winnerTechnique').textContent = (best.model_label || modelValue(best)) + ' / ' + best.module + ' / ' + best.technique; document.getElementById('winnerScore').textContent = fmt(best.metrics?.[state.sortMetric]); }} else {{ document.getElementById('winnerName').textContent = 'No combined run'; document.getElementById('winnerTechnique').textContent = 'Choose Combined techniques or All runs'; document.getElementById('winnerScore').textContent = '—'; }} }}
function sortRows() {{ state.sortMetric = document.getElementById('sortMetric').value || primaryMetric; state.direction = document.getElementById('sortDirection').value; renderSummary(); renderRows(); }}
function filterRows() {{ for (const field of filterFields) state[field] = document.getElementById(field + 'Filter').value; state.runType = document.getElementById('runTypeFilter').value; state.search = document.getElementById('searchFilter').value.toLowerCase().trim(); syncCascadedFilters(); renderSummary(); renderRows(); }}
fillSelect('metricOptions', metricKeys); document.getElementById('runTypeFilter').value = state.runType; renderHead(); syncCascadedFilters(); renderSummary(); renderRows(); for (const field of filterFields) document.getElementById(field + 'Filter').addEventListener('change', filterRows); document.getElementById('runTypeFilter').addEventListener('change', filterRows); document.getElementById('sortMetric').addEventListener('change', sortRows); document.getElementById('sortDirection').addEventListener('change', sortRows); document.getElementById('searchFilter').addEventListener('input', filterRows);
</script>
</body>
</html>
"""
    path = output_dir / "dashboard.html"
    path.write_text(document, encoding="utf-8")
    return path

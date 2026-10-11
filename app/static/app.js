const ENDPOINTS = [
  { key: "decisions", label: "Decisions API", short: "Decisions", color: "var(--c-decisions)" },
  { key: "responses_default", label: "Responses API, default", short: "Responses, default", color: "var(--c-default)" },
  { key: "responses_tuned", label: "Responses API, tuned", short: "Responses, tuned", color: "var(--c-tuned)" },
];
// What each task stands for in a product. Keyed by task name; a task without an entry shows none.
const USE_CASES = {
  intent: {
    use: "Route a customer message to the right queue",
    detail: "A support inbox, chatbot or phone menu needs one label out of many, fast, with a confidence that says when to hand over to a person.",
  },
  injection: {
    use: "Screen prompts before they reach your model",
    detail: "A yes-or-no guardrail that runs on every request, so it has to be quick and cheap, with a probability you can set a threshold on.",
  },
  relevance: {
    use: "Grade how well a passage answers a query",
    detail: "Rerank search or retrieval results, or label relevance data at scale, on an ordered scale from 0 to 3.",
  },
};
const CASE_LIMIT = 200;
const state = { tasks: [], reports: {}, tab: "summary", disagreeOnly: false, visit: 0 };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  node.append(...children.flat().filter((child) => child != null && child !== false));
  return node;
}

function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.append(...children.flat().filter((child) => child != null));
  return node;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "The request was not valid.";
    throw new Error(detail);
  }
  return body;
}

const NO_DATA = "no data";
const formatValue = (value, format) =>
  value == null ? NO_DATA : format === "percent" ? (value * 100).toFixed(1) + "%" : value.toFixed(3);
const formatMs = (value) => (value == null ? NO_DATA : Math.round(value).toLocaleString() + " ms");
const formatCost = (value) => (value == null ? NO_DATA : "$" + value.toFixed(3));

function formatMetric(metric) {
  if (!metric || metric.value == null) return NO_DATA;
  const range = metric.ci
    ? ` (${formatValue(metric.ci[0], metric.format)} to ${formatValue(metric.ci[1], metric.format)})`
    : "";
  return formatValue(metric.value, metric.format) + range;
}

function formatCounts(counts) {
  const extra = [];
  if (counts.refused) extra.push(`${counts.refused} refused`);
  if (counts.malformed) extra.push(`${counts.malformed} malformed`);
  if (counts.errored) extra.push(`${counts.errored} errors`);
  if (counts.missing) extra.push(`${counts.missing} not run`);
  return `${counts.answered} of ${counts.planned}` + (extra.length ? ` (${extra.join(", ")})` : "");
}

const cell = (text) => el("td", { class: text === NO_DATA ? "empty" : null }, text);
const swatch = (endpoint) => el("span", { class: "swatch", style: `background:${endpoint.color}` });
const endpointHeader = (endpoint) => el("th", {}, swatch(endpoint), endpoint.label);

function renderTabs() {
  const tabs = [{ name: "summary", title: "Summary" }, ...state.tasks];
  document.getElementById("tabs").replaceChildren(
    ...tabs.map((tab) =>
      el("button", { role: "tab", "aria-selected": String(state.tab === tab.name), onclick: () => show(tab.name) }, tab.title),
    ),
  );
}

async function show(tab) {
  state.tab = tab;
  state.disagreeOnly = false;
  const visit = ++state.visit;
  renderTabs();
  const view = document.getElementById("view");
  view.replaceChildren(el("p", { class: "note", style: "margin-top:40px" }, "Loading…"));
  let nodes;
  try {
    nodes = tab === "summary" ? await renderSummary() : await renderTask(tab);
  } catch (error) {
    nodes = [el("p", { class: "error", style: "margin-top:40px" }, error.message)];
  }
  if (visit !== state.visit) return; // a later click already replaced this view
  view.replaceChildren(...nodes);
  window.scrollTo(0, 0);
}

// Where the Decisions API stands against one baseline, from the same-items comparison.
function verdict(paired, format) {
  if (!paired || !paired.ci) return null;
  const size = format === "percent" ? (Math.abs(paired.difference) * 100).toFixed(1) + " pts" : Math.abs(paired.difference).toFixed(3);
  if (paired.ci[0] > 0) return { tone: "behind", text: `behind by ${size}` };
  if (paired.ci[1] < 0) return { tone: "ahead", text: `ahead by ${size}` };
  return { tone: "level", text: "level" };
}

function range(values, unit) {
  const usable = values.filter((value) => value != null && Number.isFinite(value)).map(Math.round);
  if (!usable.length) return null;
  const low = Math.min(...usable), high = Math.max(...usable);
  return (low === high ? String(low) : `${low}–${high}`) + unit;
}

function statCard(kicker, big, tone, caption) {
  return el(
    "div",
    { class: "stat rise" },
    el("div", { class: "kicker" }, kicker),
    el("div", { class: "big" + (big && tone ? " " + tone : "") + (big ? "" : " empty") }, big || NO_DATA),
    el("p", {}, caption),
  );
}

function heroStats(summary) {
  const byTask = state.tasks.map((task) => Object.fromEntries(summary.rows.filter((row) => row.task === task.name).map((row) => [row.endpoint, row])));
  const ratio = (rows, key, field) => (rows.decisions && rows[key] && rows.decisions[field] && rows[key][field] ? rows[key][field] / rows.decisions[field] : null);
  const fasterDefault = range(byTask.map((rows) => ratio(rows, "responses_default", "p50_ms")), "×");
  const fasterTuned = range(byTask.map((rows) => ratio(rows, "responses_tuned", "p50_ms")), "×");
  const saving = byTask.map((rows) => {
    const r = ratio(rows, "responses_default", "cost_per_1000");
    return r == null ? null : (1 - 1 / r) * 100;
  });
  const cheaper = saving.every((value) => value == null || value > 0) ? range(saving, "%") : null;
  const verdicts = (key) => byTask.map((rows) => (rows[key] ? verdict(rows[key].paired, rows[key].headline.format) : null)).filter(Boolean);
  const againstDefault = verdicts("responses_default"), againstTuned = verdicts("responses_tuned");
  const behind = (list) => list.filter((v) => v.tone === "behind").length;
  const ahead = [...againstDefault, ...againstTuned].filter((v) => v.tone === "ahead").length;
  const answered = summary.rows.reduce((sum, row) => sum + row.answered, 0);
  const planned = summary.rows.reduce((sum, row) => sum + row.planned, 0);
  return el(
    "div",
    { class: "stats" },
    statCard("Speed", fasterDefault, "good", fasterTuned ? `faster than the default Responses API at the median, and ${fasterTuned} faster than Responses with reasoning off.` : "faster than the default Responses API at the median."),
    statCard(
      "Quality",
      againstDefault.length ? `${behind(againstDefault)} of ${againstDefault.length}` : null,
      behind(againstDefault) ? "bad" : null,
      `tasks where it trails the default Responses API on the same items (${behind(againstTuned)} of ${againstTuned.length} against tuned). Ahead on ${ahead || "none"}.`,
    ),
    statCard("Cost", cheaper, "good", "cheaper than the default Responses API per 1,000 decisions. It bills input tokens only."),
    statCard("Reliability", planned ? `${answered.toLocaleString()}` : null, null, `of ${planned.toLocaleString()} calls came back as a usable, well-formed answer.`),
  );
}

function barRow(endpoint, share, text) {
  return el(
    "div",
    { class: "bar-row" },
    el("span", { class: "name" }, swatch(endpoint), endpoint.short),
    el("div", { class: "track" }, share == null ? null : el("div", { class: "fill", style: `width:${Math.max(0, Math.min(1, share)) * 100}%;background:${endpoint.color}` })),
    el("span", { class: "value" + (text === NO_DATA ? " empty" : "") }, text),
  );
}

function useCaseCard(task, rows, index) {
  const copy = USE_CASES[task.name];
  const byKey = Object.fromEntries(rows.map((row) => [row.endpoint, row]));
  const present = ENDPOINTS.filter((endpoint) => byKey[endpoint.key]);
  const slowest = Math.max(...present.map((e) => byKey[e.key].p50_ms || 0));
  const headline = rows[0].headline;
  const pills = present
    .filter((e) => e.key !== "decisions")
    .map((e) => {
      const v = verdict(byKey[e.key].paired, byKey[e.key].headline.format);
      return v ? el("span", { class: `pill ${v.tone}` }, `vs ${e.short}: ${v.text}`) : null;
    });
  return el(
    "article",
    { class: "usecase rise", style: `animation-delay:${index * 70}ms` },
    el("div", {}, el("span", { class: "chip accent" }, task.questions[0].type + " question")),
    el("h3", {}, task.title),
    copy ? el("p", { class: "use" }, copy.use) : null,
    copy ? el("p", { class: "note" }, copy.detail) : null,
    el(
      "div",
      {},
      el("div", { class: "block-title" }, headline.label),
      ...present.map((e) => barRow(e, byKey[e.key].headline.value, formatValue(byKey[e.key].headline.value, headline.format))),
    ),
    el(
      "div",
      {},
      el("div", { class: "block-title" }, "Median response time"),
      ...present.map((e) => barRow(e, byKey[e.key].p50_ms == null || !slowest ? null : byKey[e.key].p50_ms / slowest, formatMs(byKey[e.key].p50_ms))),
    ),
    pills.some(Boolean) ? el("div", { class: "verdicts", title: "Decisions API quality on the same items; a gap counts only when its 95% interval excludes zero" }, pills) : null,
    el("button", { class: "ghost", type: "button", onclick: () => show(task.name) }, "Explore and try it →"),
  );
}

async function renderSummary() {
  const summary = await getJson("/api/summary");
  const rows = [];
  for (const task of state.tasks) {
    const taskRows = summary.rows.filter((row) => row.task === task.name);
    taskRows.forEach((row, index) => {
      const endpoint = ENDPOINTS.find((e) => e.key === row.endpoint);
      rows.push(
        el(
          "tr",
          {},
          index === 0 ? el("td", { rowspan: taskRows.length }, el("strong", {}, row.title), el("span", { class: "sub" }, row.headline.label)) : null,
          el("td", {}, swatch(endpoint), endpoint.label),
          cell(formatMetric(row.headline)),
          cell(formatMs(row.p50_ms)),
          cell(formatMs(row.p95_ms)),
          cell(formatCost(row.cost_per_1000)),
          el("td", {}, `${row.answered} of ${row.planned}`),
        ),
      );
    });
  }
  const prices = summary.prices.per_million;
  const calls = summary.rows.reduce((sum, row) => sum + row.planned, 0);
  const cards = state.tasks
    .map((task) => [task, summary.rows.filter((row) => row.task === task.name)])
    .filter(([, taskRows]) => taskRows.length)
    .map(([task, taskRows], index) => useCaseCard(task, taskRows, index));
  return [
    el(
      "section",
      { class: "hero rise" },
      el(
        "div",
        { class: "eyebrow" },
        el("span", { class: "chip accent" }, "Benchmark"),
        el("span", { class: "chip" }, summary.model),
        el("span", { class: "chip" }, `${state.tasks.length} use cases`),
        el("span", { class: "chip" }, `${calls.toLocaleString()} calls`),
      ),
      el("h1", {}, "How good is the ", el("span", { class: "grad" }, "Decisions API"), "?"),
      el(
        "p",
        { class: "lede" },
        "Three jobs it should be best at, each asked three ways of the same model: typed decisions, the Responses API, and the Responses API with reasoning off.",
      ),
      heroStats(summary),
    ),
    el("h2", {}, "Use cases"),
    el("p", { class: "note" }, "One task per Decisions question type. Bars show the headline quality metric and the median response time."),
    el("div", { class: "usecases" }, cards),
    el("h2", {}, "All the numbers"),
    el(
      "div",
      { class: "panel scroll" },
      el(
        "table",
        {},
        el("thead", {}, el("tr", {}, ...["Task", "Endpoint", "Headline result", "Median time", "95th percentile", "Cost per 1,000", "Answered"].map((h) => el("th", {}, h)))),
        el("tbody", {}, rows),
      ),
    ),
    el(
      "p",
      { class: "note" },
      `Ranges are 95% intervals. Cost uses reported token counts at $${prices.input.toFixed(2)} input, ` +
        `$${prices.cached_input.toFixed(2)} cached input and $${prices.output.toFixed(2)} output per million tokens ` +
        `(read ${summary.prices.read_on}). The Decisions API bills input tokens only.`,
    ),
  ];
}

async function loadReport(name) {
  if (!state.reports[name]) state.reports[name] = await getJson(`/api/results/${name}`);
  return state.reports[name];
}

async function renderTask(name) {
  const task = state.tasks.find((t) => t.name === name);
  const question = task.questions[0];
  const copy = USE_CASES[task.name];
  const intro = [
    el(
      "section",
      { class: "task-head rise" },
      el("span", { class: "chip accent" }, question.type + " question"),
      el("h1", {}, task.title),
      copy ? el("p", { class: "use" }, copy.use + ". " + copy.detail) : null,
      el("p", { class: "question" }, el("span", {}, "The question every endpoint is asked"), question.instructions),
    ),
    playground(task),
  ];
  let report;
  try {
    report = await loadReport(name);
  } catch (error) {
    return [...intro, el("h2", {}, "Results"), el("p", { class: "empty" }, error.message)];
  }
  const dataset = report.dataset ? `${report.dataset.name}. ${report.dataset.sampling || ""}` : "";
  return [
    ...intro,
    el("h2", {}, "Results"),
    el("p", { class: "note" }, dataset),
    el("div", { class: "panel scroll" }, comparisonTable(report)),
    el(
      "div",
      { class: "charts" },
      el("div", { class: "panel" }, el("strong", {}, "Response time"), latencyChart(report), legend(report)),
      el("div", { class: "panel" }, el("strong", {}, "Calibration"), calibrationChart(report), legend(report)),
    ),
    el("h2", {}, "By group"),
    el("div", { class: "panel scroll" }, breakdownTable(report)),
    el("h2", {}, "Cases"),
    casesPanel(report),
  ];
}

function legend(report) {
  return el("div", { class: "legend" }, ...presentEndpoints(report).map((e) => el("span", {}, swatch(e), e.label)));
}

function formatPaired(report, endpoint) {
  if (endpoint.key === "decisions") return "baseline";
  const paired = (report.paired || {})[endpoint.key];
  if (!paired || !paired.ci) return NO_DATA;
  const percent = report.endpoints[endpoint.key].metrics[report.headline].format === "percent";
  const show = (value) => (value >= 0 ? "+" : "−") + (percent ? (Math.abs(value) * 100).toFixed(1) : Math.abs(value).toFixed(3));
  return `${show(paired.difference)}${percent ? " pts" : ""} (${show(paired.ci[0])} to ${show(paired.ci[1])})`;
}

function presentEndpoints(report) {
  return ENDPOINTS.filter((endpoint) => report.endpoints[endpoint.key]);
}

function comparisonTable(report) {
  const endpoints = presentEndpoints(report);
  const row = (label, value) => el("tr", {}, el("td", {}, label), ...endpoints.map((e) => cell(value(report.endpoints[e.key]))));
  const metricKeys = Object.keys(report.endpoints[endpoints[0].key].metrics);
  return el(
    "table",
    {},
    el("thead", {}, el("tr", {}, el("th", {}, ""), ...endpoints.map(endpointHeader))),
    el(
      "tbody",
      {},
      ...metricKeys.map((key) => row(report.endpoints[endpoints[0].key].metrics[key].label, (s) => formatMetric(s.metrics[key]))),
      el(
        "tr",
        {},
        el("td", {}, `${report.endpoints[endpoints[0].key].metrics[report.headline].label}, gap to the Decisions API`, el("span", { class: "sub" }, "same items; real when the range excludes zero")),
        ...endpoints.map((e) => cell(formatPaired(report, e))),
      ),
      row("Calibration error", (s) => formatValue(s.calibration.error, "number")),
      row("Median response time", (s) => formatMs(s.latency_ms.p50)),
      row("95th-percentile response time", (s) => formatMs(s.latency_ms.p95)),
      row("Mean input tokens", (s) => (s.tokens.input == null ? NO_DATA : Math.round(s.tokens.input).toLocaleString())),
      row("Mean output tokens", (s) => (s.tokens.output == null ? NO_DATA : Math.round(s.tokens.output).toLocaleString())),
      row("Cost per 1,000 decisions", (s) => formatCost(s.cost_per_1000)),
      row("Answered", (s) => formatCounts(s.counts)),
    ),
  );
}

function bucketLabel(bucket) {
  const short = (ms) => (ms >= 1000 ? ms / 1000 + "s" : String(ms));
  return bucket.high == null ? short(bucket.low) + "+" : short(bucket.low);
}

function latencyChart(report) {
  const endpoints = presentEndpoints(report).filter((e) => report.endpoints[e.key].latency_ms.p50 != null);
  if (!endpoints.length) return el("p", { class: "empty" }, NO_DATA);
  const W = 480, H = 220, left = 34, bottom = 30, top = 8;
  const histograms = endpoints.map((e) => report.endpoints[e.key].latency_ms.histogram);
  const shares = histograms.map((buckets) => {
    const total = buckets.reduce((sum, b) => sum + b.n, 0);
    return buckets.map((b) => b.n / total);
  });
  const peak = Math.max(...shares.flat());
  const buckets = histograms[0];
  const slot = (W - left) / buckets.length;
  const bar = (slot - 4) / endpoints.length;
  const plotHeight = H - bottom - top;
  const chart = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share of calls by response time" });
  chart.append(
    svgEl("line", { class: "axis", x1: left, y1: H - bottom, x2: W, y2: H - bottom }),
    svgEl("text", { x: 0, y: top + 8 }, Math.round(peak * 100) + "%"),
    svgEl("text", { x: W, y: H - 2, "text-anchor": "end" }, "response time, from (ms)"),
  );
  buckets.forEach((bucket, b) => {
    chart.append(svgEl("text", { x: left + b * slot + slot / 2, y: H - bottom + 12, "text-anchor": "middle" }, bucketLabel(bucket)));
    endpoints.forEach((endpoint, e) => {
      const height = (shares[e][b] / peak) * plotHeight;
      chart.append(
        svgEl("rect", { x: left + b * slot + 2 + e * bar, y: H - bottom - height, width: Math.max(bar - 1, 1), height, rx: 2, fill: endpoint.color }),
      );
    });
  });
  return chart;
}

function calibrationChart(report) {
  const endpoints = presentEndpoints(report).filter((e) => report.endpoints[e.key].calibration.error != null);
  if (!endpoints.length) return el("p", { class: "empty" }, NO_DATA);
  const first = report.endpoints[endpoints[0].key].calibration;
  const W = 480, H = 220, left = 34, bottom = 30, top = 8;
  const span = first.high - first.low;
  const x = (value) => left + ((value - first.low) / span) * (W - left - 8);
  const y = (value) => H - bottom - ((value - first.low) / span) * (H - bottom - top);
  const chart = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${first.y_label} against ${first.x_label}` });
  chart.append(
    svgEl("line", { class: "axis", x1: left, y1: H - bottom, x2: W - 8, y2: H - bottom }),
    svgEl("line", { class: "axis", x1: left, y1: top, x2: left, y2: H - bottom }),
    svgEl("line", { class: "axis", x1: x(first.low), y1: y(first.low), x2: x(first.high), y2: y(first.high), "stroke-dasharray": "4 4" }),
    svgEl("text", { x: W - 8, y: H - 2, "text-anchor": "end" }, first.x_label),
    svgEl("text", { x: left + 6, y: top + 8 }, first.y_label),
    svgEl("text", { x: left, y: H - bottom + 12, "text-anchor": "middle" }, String(first.low)),
    svgEl("text", { x: W - 8, y: H - bottom + 12, "text-anchor": "end" }, String(first.high)),
  );
  for (const endpoint of endpoints) {
    const points = report.endpoints[endpoint.key].calibration.bins.filter((bin) => bin.n > 0);
    chart.append(
      svgEl("polyline", { points: points.map((p) => `${x(p.mean_x)},${y(p.mean_y)}`).join(" "), fill: "none", stroke: endpoint.color, "stroke-width": 2 }),
      ...points.map((p) => svgEl("circle", { cx: x(p.mean_x), cy: y(p.mean_y), r: 3, fill: endpoint.color })),
    );
  }
  return chart;
}

function breakdownTable(report) {
  const endpoints = presentEndpoints(report);
  const groups = new Map();
  for (const endpoint of endpoints) {
    for (const group of report.endpoints[endpoint.key].breakdown) {
      if (!groups.has(group.group)) groups.set(group.group, {});
      groups.get(group.group)[endpoint.key] = group;
    }
  }
  if (!groups.size) return el("p", { class: "empty" }, NO_DATA);
  const names = [...groups.keys()].sort();
  return el(
    "table",
    {},
    el("thead", {}, el("tr", {}, el("th", {}, "Group"), ...endpoints.map(endpointHeader))),
    el(
      "tbody",
      {},
      ...names.map((name) =>
        el(
          "tr",
          {},
          el("td", { class: "text" }, name),
          ...endpoints.map((endpoint) => {
            const group = groups.get(name)[endpoint.key];
            return cell(group ? `${group.correct} of ${group.n} correct` : NO_DATA);
          }),
        ),
      ),
    ),
  );
}

function answerCell(answer) {
  if (!answer) return el("td", { class: "empty" }, "not run");
  if (answer.status !== "ok") return el("td", { class: "wrong" }, answer.status);
  const detail = answer.probability == null ? "" : ` (${(answer.probability * 100).toFixed(0)}%)`;
  return el("td", { class: answer.correct ? "right" : "wrong" }, String(answer.label) + detail);
}

function casesPanel(report) {
  const endpoints = presentEndpoints(report);
  const body = el("tbody");
  const count = el("p", { class: "note" });
  const disagree = (c) => new Set(endpoints.map((e) => (c.answers[e.key] ? String(c.answers[e.key].label) : "none"))).size > 1;
  const fill = () => {
    const cases = state.disagreeOnly ? report.cases.filter(disagree) : report.cases;
    count.textContent = `Showing ${Math.min(cases.length, CASE_LIMIT)} of ${cases.length} cases.`;
    body.replaceChildren(
      ...cases.slice(0, CASE_LIMIT).map((c) =>
        el(
          "tr",
          {},
          el("td", { class: "text" }, ...report.fields.map(([key, label]) => el("div", {}, report.fields.length > 1 ? el("strong", {}, label + ": ") : null, c.fields[key]))),
          el("td", {}, String(c.label)),
          ...endpoints.map((endpoint) => answerCell(c.answers[endpoint.key])),
        ),
      ),
    );
  };
  fill();
  return el(
    "div",
    { class: "panel" },
    el(
      "label",
      { class: "check" },
      el("input", { type: "checkbox", onchange: (event) => { state.disagreeOnly = event.target.checked; fill(); } }),
      "Only cases where the endpoints disagree",
    ),
    count,
    el(
      "div",
      { class: "scroll" },
      el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Input"), el("th", {}, "Correct answer"), ...endpoints.map(endpointHeader))), body),
    ),
  );
}

function answerCard(endpoint, answer) {
  const card = el("div", { class: "answer", style: `border-top-color:${endpoint.color}` }, el("h4", {}, swatch(endpoint), endpoint.label));
  if (!answer) return card;
  if (answer.error) {
    card.append(el("p", { class: "error" }, answer.error));
    return card;
  }
  card.append(
    el("div", { class: "big" }, answer.status === "ok" ? String(answer.label) : answer.status),
    el("div", { class: "speed", style: `color:${endpoint.color}` }, formatMs(answer.latency_ms)),
  );
  if (answer.probability != null) {
    card.append(el("div", { class: "track" }, el("div", { class: "fill", style: `width:${Math.max(0, Math.min(1, answer.probability)) * 100}%;background:${endpoint.color}` })));
  }
  const facts = [];
  if (answer.probability != null) facts.push(["Probability", (answer.probability * 100).toFixed(1) + "%"]);
  if (answer.expected != null) facts.push(["Expected grade", answer.expected.toFixed(2)]);
  if (answer.ranked && answer.ranked.length > 1) facts.push(["Next most likely", answer.ranked.slice(1).join(", ")]);
  facts.push(["Tokens in / out", `${answer.input_tokens ?? "unknown"} / ${answer.output_tokens ?? "unknown"}`]);
  card.append(el("dl", {}, ...facts.flatMap(([term, value]) => [el("dt", {}, term), el("dd", {}, value)])));
  return card;
}

function playground(task) {
  const inputs = Object.fromEntries(task.fields.map(([key]) => [key, el("textarea", { id: `field-${key}`, required: true })]));
  const output = el("div", { class: "answers" });
  const message = el("span", { class: "error", role: "alert" });
  const submit = el("button", { class: "primary", type: "submit" }, "Run on all three");
  const setExample = (example) => task.fields.forEach(([key]) => { inputs[key].value = example[key]; });

  async function run(event) {
    event.preventDefault();
    message.textContent = "";
    submit.disabled = true;
    submit.textContent = "Running…";
    try {
      const fields = Object.fromEntries(task.fields.map(([key]) => [key, inputs[key].value]));
      const result = await getJson(`/api/try/${task.name}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fields }),
      });
      output.replaceChildren(...ENDPOINTS.filter((e) => result.answers[e.key]).map((e) => answerCard(e, result.answers[e.key])));
    } catch (error) {
      output.replaceChildren();
      message.textContent = error.message;
    } finally {
      submit.disabled = false;
      submit.textContent = "Run on all three";
    }
  }

  return el(
    "form",
    { class: "panel", onsubmit: run },
    el("strong", {}, "Try it live"),
    el("p", { class: "note" }, "Your input goes to all three endpoints at once. Watch the response times."),
    ...task.fields.flatMap(([key, label]) => [el("label", { for: `field-${key}` }, label), inputs[key]]),
    el(
      "div",
      { class: "actions" },
      submit,
      ...task.examples.map((example, index) => el("button", { class: "ghost", type: "button", onclick: () => setExample(example) }, `Example ${index + 1}`)),
      message,
    ),
    output,
  );
}

async function init() {
  try {
    state.tasks = await getJson("/api/tasks");
  } catch (error) {
    document.getElementById("view").replaceChildren(el("p", { class: "error" }, error.message));
    return;
  }
  show("summary");
}

init();

const ENDPOINTS = [
  { key: "decisions", label: "Decisions API", color: "var(--c-decisions)" },
  { key: "responses_default", label: "Responses API, default", color: "var(--c-default)" },
  { key: "responses_tuned", label: "Responses API, tuned", color: "var(--c-tuned)" },
];
const CASE_LIMIT = 200;
const state = { tasks: [], reports: {}, tab: "summary", disagreeOnly: false };

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
  renderTabs();
  const view = document.getElementById("view");
  view.replaceChildren(el("p", { class: "note" }, "Loading…"));
  try {
    view.replaceChildren(...(tab === "summary" ? await renderSummary() : await renderTask(tab)));
  } catch (error) {
    view.replaceChildren(el("p", { class: "error" }, error.message));
  }
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
          index === 0 ? el("td", { rowspan: taskRows.length }, el("strong", {}, row.title), el("br"), row.headline.label) : null,
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
  return [
    el("h2", {}, "All tasks at a glance"),
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
  const intro = [
    el("h2", {}, task.title),
    el("p", { class: "note" }, `One ${question.type} question: “${question.instructions}”`),
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
      el("div", { class: "panel" }, el("strong", {}, "Response time"), latencyChart(report)),
      el("div", { class: "panel" }, el("strong", {}, "Calibration"), calibrationChart(report)),
    ),
    el("h2", {}, "By group"),
    el("div", { class: "panel scroll" }, breakdownTable(report)),
    el("h2", {}, "Cases"),
    casesPanel(report),
  ];
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
        svgEl("rect", { x: left + b * slot + 2 + e * bar, y: H - bottom - height, width: Math.max(bar - 1, 1), height, fill: endpoint.color }),
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
      {},
      el("input", { type: "checkbox", onchange: (event) => { state.disagreeOnly = event.target.checked; fill(); } }),
      " Only cases where the endpoints disagree",
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
  const card = el("div", { class: "answer" }, el("h3", {}, swatch(endpoint), endpoint.label));
  if (!answer) return card;
  if (answer.error) {
    card.append(el("p", { class: "error" }, answer.error));
    return card;
  }
  card.append(el("div", { class: "big" }, answer.status === "ok" ? String(answer.label) : answer.status));
  const facts = [];
  if (answer.probability != null) facts.push(["Probability", (answer.probability * 100).toFixed(1) + "%"]);
  if (answer.expected != null) facts.push(["Expected grade", answer.expected.toFixed(2)]);
  if (answer.ranked && answer.ranked.length > 1) facts.push(["Next most likely", answer.ranked.slice(1).join(", ")]);
  facts.push(["Response time", formatMs(answer.latency_ms)]);
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

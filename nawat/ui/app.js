/* Nawāt — the working area behind the instrument frame.
   Plain JS against the control-plane API. No framework: everything on screen
   is either a reading or a control, and both are cheap to draw by hand. */

"use strict";

/* ---- API client ---------------------------------------------------------- */

const TOKEN_KEY = "nawat-token";

function authHeaders() {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: "Bearer " + token } : {};
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(options.headers || {}) },
  });
  if (response.status === 401) {
    document.getElementById("gate").classList.add("open");
    throw new ApiError({ cause: "A valid token is required.", remedy: "" }, 401);
  }
  if (!response.ok) {
    let body = {};
    try { body = await response.json(); } catch {}
    throw new ApiError(body, response.status);
  }
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

class ApiError extends Error {
  constructor(body, status) {
    const detail = body.cause || body.detail || "The control plane returned an error.";
    super(body.remedy ? `${detail} ${body.remedy}` : detail);
    this.status = status;
  }
}

/* Server-sent events over fetch, so the auth header travels too. */
async function sse(path, onData, onEvent, signal) {
  const response = await fetch(path, { headers: authHeaders(), signal });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (event === "message") onData(data.join("\n"));
      else if (onEvent) onEvent(event, data.join("\n"));
    }
  }
}

/* ---- tiny DOM helpers ---------------------------------------------------- */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

function bytes(count) {
  if (count === 0) return "0 B";
  const units = [["PB", 1e15], ["TB", 1e12], ["GB", 1e9], ["MB", 1e6], ["KB", 1e3]];
  for (const [unit, scale] of units) {
    if (Math.abs(count) >= scale) {
      const value = count / scale;
      return (value < 100 ? value.toFixed(1) : Math.round(value)) + " " + unit;
    }
  }
  return Math.round(count) + " B";
}

function age(seconds) {
  seconds = Math.max(0, seconds);
  if (seconds < 60) return Math.floor(seconds) + "s";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m";
  if (seconds < 86400) return Math.floor(seconds / 3600) + "h";
  return Math.floor(seconds / 86400) + "d";
}

function notice(kind, text) {
  return el("p", { class: "notice " + kind }, text);
}

function stateSpan(state) {
  return el("span", { class: "state " + state },
    ["running", "staging", "publishing", "starting"].includes(state) ? el("i", { class: "dot" }) : null,
    state);
}

/* ---- the trace ----------------------------------------------------------- */
/* A gold line on a fine graticule with real axis ticks. No container, no
   fill, no rounded anything. Comparison series draw beneath in verdigris. */

const PALETTE = { gold: "#C9A227", verdigris: "#4E8C86", muted: "#8494A6", rubric: "#C2492E",
                  graticule: "rgba(132,148,166,0.10)", axis: "rgba(132,148,166,0.35)" };
const COMPARE_COLORS = ["#4E8C86", "#7A6FA0", "#A08C4E", "#5E7CA6"];

class Trace {
  constructor(canvas) {
    this.canvas = canvas;
    this.series = [];   // {label, points: [{step, value}], color, live}
    this.events = [];   // {step, event}
    this.smoothing = 0;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  set(series, events = []) { this.series = series; this.events = events; this.draw(); }

  smoothed(points) {
    if (this.smoothing <= 0 || points.length < 3) return points;
    const alpha = 1 - this.smoothing * 0.92;
    let value = points[0].value;
    return points.map(p => ({ step: p.step, value: (value = alpha * p.value + (1 - alpha) * value) }));
  }

  draw() {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width) return;
    canvas.width = width * dpr; canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const pad = { left: 56, right: 12, top: 10, bottom: 26 };
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    const all = this.series.flatMap(s => s.points);
    if (!all.length) {
      ctx.fillStyle = PALETTE.muted;
      ctx.font = "12px 'IBM Plex Mono', monospace";
      ctx.fillText("no readings yet", pad.left, height / 2);
      return;
    }

    let x0 = Math.min(...all.map(p => p.step)), x1 = Math.max(...all.map(p => p.step));
    let y0 = Math.min(...all.map(p => p.value)), y1 = Math.max(...all.map(p => p.value));
    if (x0 === x1) x1 = x0 + 1;
    if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
    const ySpan = y1 - y0; y0 -= ySpan * 0.06; y1 += ySpan * 0.06;

    const X = s => pad.left + ((s - x0) / (x1 - x0)) * plotW;
    const Y = v => pad.top + (1 - (v - y0) / (y1 - y0)) * plotH;

    // graticule + real ticks
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.lineWidth = 1;
    const xTicks = niceTicks(x0, x1, Math.max(3, Math.floor(plotW / 90)));
    const yTicks = niceTicks(y0, y1, Math.max(3, Math.floor(plotH / 48)));
    for (const t of yTicks) {
      ctx.strokeStyle = PALETTE.graticule;
      line(ctx, pad.left, Y(t), width - pad.right, Y(t));
      ctx.fillStyle = PALETTE.muted;
      ctx.textAlign = "right";
      ctx.fillText(fmtTick(t), pad.left - 8, Y(t) + 3);
    }
    for (const t of xTicks) {
      ctx.strokeStyle = PALETTE.graticule;
      line(ctx, X(t), pad.top, X(t), height - pad.bottom);
      ctx.fillStyle = PALETTE.muted;
      ctx.textAlign = "center";
      ctx.fillText(fmtTick(t), X(t), height - 8);
    }
    ctx.strokeStyle = PALETTE.axis;
    line(ctx, pad.left, pad.top, pad.left, height - pad.bottom);
    line(ctx, pad.left, height - pad.bottom, width - pad.right, height - pad.bottom);

    // event marks: short vermilion strokes on the baseline
    for (const mark of this.events) {
      ctx.strokeStyle = PALETTE.rubric;
      line(ctx, X(mark.step), height - pad.bottom, X(mark.step), height - pad.bottom - 8);
    }

    // comparison series beneath, live series last and on top
    const ordered = [...this.series].sort((a, b) => (a.live ? 1 : 0) - (b.live ? 1 : 0));
    for (const s of ordered) {
      if (!s.points.length) continue;
      if (this.smoothing > 0 && s.live) this.path(ctx, s.points, X, Y, s.color, 0.25, 1);
      this.path(ctx, this.smoothed(s.points), X, Y, s.color, 1, s.live ? 1.5 : 1);
    }
  }

  path(ctx, points, X, Y, color, alpha, widthPx) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = widthPx;
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(X(p.step), Y(p.value)) : ctx.moveTo(X(p.step), Y(p.value))));
    ctx.stroke();
    ctx.restore();
  }
}

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath(); ctx.moveTo(x1 + 0.5, y1 + 0.5); ctx.lineTo(x2 + 0.5, y2 + 0.5); ctx.stroke();
}

function niceTicks(low, high, count) {
  const span = high - low;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const candidates = [step, step * 2, step * 2.5, step * 5, step * 10];
  const chosen = candidates.find(c => span / c <= count) || step * 10;
  const ticks = [];
  for (let t = Math.ceil(low / chosen) * chosen; t <= high + 1e-9; t += chosen) ticks.push(t);
  return ticks;
}

function fmtTick(value) {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return Math.round(value).toLocaleString("en");
  if (magnitude < 0.01) return value.toExponential(1);
  return +value.toFixed(magnitude < 1 ? 3 : 2) + "";
}

function traceBlock() {
  const canvas = el("canvas", { class: "trace" });
  const trace = new Trace(canvas);
  const legend = el("span", { class: "legend" });
  const slider = el("input", { type: "range", min: 0, max: 100, value: 0,
    oninput: () => { trace.smoothing = slider.value / 100; trace.draw(); } });
  const wrap = el("div", { class: "trace-wrap" }, canvas,
    el("div", { class: "trace-controls" }, "smoothing", slider, legend));
  wrap.trace = trace;
  wrap.setLegend = entries => {
    legend.replaceChildren(...entries.map(e =>
      el("span", {}, el("i", { class: "key", style: "background:" + e.color }), e.label)));
  };
  return wrap;
}

/* ---- views --------------------------------------------------------------- */

const main = document.getElementById("main");
let disposeCurrent = () => {};

const views = {

  /* Storage — occupancy, artifacts, and the controls the researcher owns. */
  async storage() {
    const status = await api("/cache");
    const artifacts = await api("/cache/artifacts");
    const frame = el("div", {},
      el("p", { class: "eyebrow" }, "Local disk"),
      el("h1", {}, "Storage"),
      el("section", { class: "panel" },
        el("table", {}, el("tbody", {},
          measureRow("Cache", `${bytes(status.used)} of ${bytes(status.ceiling)} (${Math.round(status.fraction * 100)}%)`),
          measureRow("Disk free", bytes(status.disk_free)),
          measureRow("Kept on disk", bytes(status.pinned_bytes)),
          measureRow("In use", bytes(status.leased_bytes)),
          measureRow("Only on this disk", bytes(status.unreplicated_bytes),
            status.unreplicated_bytes > 0 ? "alarm" : ""))),
        el("div", { class: "actions" },
          el("button", { onclick: () => act(frame, api("/cache/free", { method: "POST", body: JSON.stringify({}) })
            .then(r => `Freed ${bytes(r.freed)}.`)) }, "Free space")),
      ),
      el("section", { class: "panel" },
        el("p", { class: "eyebrow" }, "On disk"),
        artifacts.length ? artifactTable(frame, artifacts)
          : el("p", { class: "empty" }, "Nothing is cached locally. Stage a model from the ",
              el("a", { href: "#registry" }, "registry"), ", or submit a run.")));
    return frame;

    function artifactTable(host, rows) {
      return el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Key"), el("th", { class: "num" }, "Size"), el("th", {}, "Last use"),
          el("th", {}, "Held"), el("th", {}, "In store"), el("th", {}, ""))),
        el("tbody", {}, rows.map(a => el("tr", {},
          el("td", {}, a.key),
          el("td", { class: "num" }, a.bytes_human),
          el("td", { class: "muted" }, age(Date.now() / 1000 - a.last_used)),
          el("td", { class: "muted" }, [a.pinned ? "kept" : "", ...a.holders].filter(Boolean).join(", ") || "—"),
          el("td", { class: a.replicated ? "muted" : "" },
            a.replicated ? "yes" : el("span", { class: "state failed" }, "no")),
          el("td", {},
            el("button", { class: "quiet", onclick: () => act(host,
              api(`/cache/${a.key}/keep`, { method: a.pinned ? "DELETE" : "POST" })
                .then(() => a.pinned ? `${a.key} may be freed again.` : `Keeping ${a.key} on disk.`)) },
              a.pinned ? "Release" : "Keep on disk"),
            el("button", { class: "quiet danger", onclick: () => act(host,
              api(`/cache/${a.key}`, { method: "DELETE" }).then(r => `Removed, freeing ${r.freed_human}.`)) },
              "Remove"))))));
    }
  },

  /* Registry — object storage is truth; this is what it holds. */
  async registry() {
    const entries = await api("/registry");
    const frame = el("div", {},
      el("p", { class: "eyebrow" }, "Object storage"),
      el("h1", {}, "Registry"),
      el("section", { class: "panel" },
        entries.length ? el("table", {},
          el("thead", {}, el("tr", {}, el("th", {}, "Key"), el("th", {}, "Local"), el("th", {}, ""))),
          el("tbody", {}, entries.map(e => el("tr", {},
            el("td", {}, e.key),
            el("td", { class: "muted" }, e.cached ? "cached" : "—"),
            el("td", {}, e.cached ? null :
              el("button", { class: "quiet", onclick: () => act(frame,
                api(`/cache/${e.key}/resolve`, { method: "POST" }).then(r => `Staged to ${r.path}.`)) },
                "Stage"))))))
        : el("p", { class: "empty" }, "Object storage holds nothing yet. Publish a run, or seed a model with ",
            el("span", { class: "mono" }, "nawat resolve models/<repo>"), ".")));
    return frame;
  },

  /* Runs — history; click through to the record and its trace. */
  async runs() {
    const records = await api("/runs");
    return el("div", {},
      el("p", { class: "eyebrow" }, "Experiments"),
      el("h1", {}, "Runs"),
      el("section", { class: "panel" },
        records.length ? el("table", {},
          el("thead", {}, el("tr", {},
            el("th", {}, "Run"), el("th", {}, "State"), el("th", {}, "Script"),
            el("th", {}, "Model"), el("th", { class: "num" }, "Took"), el("th", { class: "num" }, "Artifacts"))),
          el("tbody", {}, records.map(r => el("tr", { class: "selectable",
              onclick: () => { location.hash = "#run/" + r.id; } },
            el("td", {}, r.id),
            el("td", {}, stateSpan(r.state)),
            el("td", { class: "muted" }, r.spec.script),
            el("td", { class: "muted" }, r.spec.model || "—"),
            el("td", { class: "num muted" }, r.duration ? age(r.duration) : "—"),
            el("td", { class: "num muted" }, String(r.artifacts.length))))))
        : el("p", { class: "empty" }, "No runs yet. ", el("a", { href: "#submit" }, "Submit one"), ".")));
  },

  /* One run — the trace is the reading; the log is beneath it. */
  async run(runId) {
    let record = await api(`/runs/${runId}`);
    const traceEl = traceBlock();
    const logEl = el("pre", { class: "log" }, "");
    const header = el("div", {});
    const evalPanel = el("div", {});
    const controller = new AbortController();
    disposeCurrent = () => controller.abort();

    const frame = el("div", {},
      el("p", { class: "eyebrow" }, "Run record"),
      el("h1", {}, runId, " ", el("small", { id: "run-state" })),
      header,
      el("section", { class: "panel" }, el("p", { class: "eyebrow" }, "Trace"), traceEl),
      el("section", { class: "panel" }, evalPanel),
      el("section", { class: "panel" }, el("p", { class: "eyebrow" }, "Log"), logEl));

    const renderHeader = () => {
      header.replaceChildren(el("table", {}, el("tbody", {},
        measureRow("Script", record.spec.script),
        record.spec.model ? measureRow("Model", record.spec.model) : null,
        ...record.spec.datasets.map(d => measureRow("Dataset", d)),
        ...Object.entries(record.spec.params).map(([k, v]) => measureRow(k, v)),
        record.spec.notes ? measureRow("Notes", record.spec.notes) : null,
        ...(record.description ? [measureRow("Description", record.description)] : []),
        ...record.artifacts.map(a => measureRow("Artifact", a)),
        record.error ? measureRow("Error", record.error, "alarm") : null)),
        !["succeeded", "failed", "cancelled"].includes(record.state)
          ? el("div", { class: "actions" }, el("button", { class: "danger",
              onclick: () => act(frame, api(`/runs/${runId}/cancel`, { method: "POST" }).then(() => "Cancelled.")) },
              "Cancel run"))
          : el("div", { class: "actions" },
              el("button", { onclick: () => { location.hash = "#compare/" + runId; } }, "Compare"),
              record.state === "failed"
                ? el("button", { onclick: () => { location.hash = "#agent/" + runId; } }, "Diagnose") : null,
              record.artifacts.some(a => a.endsWith("/adapter"))
                ? el("button", { onclick: () => { location.hash = "#serve/" + runId; } }, "Test adapter") : null));
      document.getElementById("run-state")?.replaceChildren(stateSpan(record.state));
    };
    renderHeader();

    const points = [];
    const redraw = () => {
      const seriesPoints = points.filter(p => "loss" in p).map(p => ({ step: p.step ?? 0, value: p.loss }));
      const marks = points.filter(p => p.event).map(p => ({ step: p.step ?? 0, event: p.event }));
      const live = !["succeeded", "failed", "cancelled"].includes(record.state);
      traceEl.trace.set([{ label: "loss", points: seriesPoints, color: live ? PALETTE.gold : PALETTE.verdigris, live }], marks);
      traceEl.setLegend([{ label: "loss — " + runId, color: live ? PALETTE.gold : PALETTE.verdigris }]);
      renderEval(evalPanel, points);
    };

    sse(`/runs/${runId}/metrics/stream`, data => { points.push(JSON.parse(data)); redraw(); },
      async () => { record = await api(`/runs/${runId}`); renderHeader(); redraw(); },
      controller.signal).catch(() => {});
    sse(`/runs/${runId}/log/stream`, dataLine => {
      logEl.textContent += dataLine + "\n";
      logEl.scrollTop = logEl.scrollHeight;
    }, null, controller.signal).catch(() => {});

    redraw();
    return frame;
  },

  /* Submit — validated before GPU time is spent. */
  async submit() {
    const [scripts, registry] = await Promise.all([api("/scripts"), api("/registry")]);
    const models = registry.filter(e => e.key.startsWith("models/"));
    const datasets = registry.filter(e => e.key.startsWith("datasets/"));

    const scriptSel = select(scripts.map(s => s.path), true);
    const modelSel = select(["", ...models.map(m => m.key)]);
    const datasetSel = select(["", ...datasets.map(d => d.key)]);
    const paramsIn = el("textarea", { rows: 3, placeholder: "learning_rate=2e-4\nmax_steps=60", spellcheck: "false" });
    const notesIn = el("input", { type: "text" });
    const out = el("div", {});

    const frame = el("div", {},
      el("p", { class: "eyebrow" }, "New experiment"),
      el("h1", {}, "Submit a run"),
      el("section", { class: "panel" },
        scripts.length ? null : el("p", { class: "empty" },
          "The workspace has no training scripts. Put a ", el("span", { class: "mono" }, ".py"),
          " or notebook in the workspace directory first."),
        el("label", {}, "Training script"), scriptSel,
        el("div", { class: "row" },
          el("div", {}, el("label", {}, "Base model"), modelSel),
          el("div", {}, el("label", {}, "Dataset"), datasetSel)),
        el("label", {}, "Parameters — one name=value per line"), paramsIn,
        el("label", {}, "Notes — why this run exists"), notesIn,
        el("div", { class: "actions" }, el("button", { onclick: submitRun }, "Submit run")),
        out));
    return frame;

    async function submitRun() {
      const params = {};
      for (const lineText of paramsIn.value.split("\n")) {
        const trimmed = lineText.trim();
        if (!trimmed) continue;
        const eq = trimmed.indexOf("=");
        if (eq < 1) { out.replaceChildren(notice("error", `"${trimmed}" is not a parameter. Write it as name=value.`)); return; }
        params[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
      }
      try {
        const record = await api("/runs", { method: "POST", body: JSON.stringify({
          script: scriptSel.value,
          model: modelSel.value || null,
          datasets: datasetSel.value ? [datasetSel.value] : [],
          params, notes: notesIn.value,
        }) });
        location.hash = "#run/" + record.id;
      } catch (error) { out.replaceChildren(notice("error", error.message)); }
    }
  },

  /* Serve — session control, adapter loading, chat with image input. */
  async serve(preferRun) {
    const registry = await api("/registry");
    let session = await api("/sessions/current");
    const models = registry.filter(e => e.key.startsWith("models/")).map(e => e.key);
    const adapters = registry.filter(e => /^runs\/.+\/adapter$/.test(e.key)).map(e => e.key);

    const sessionPanel = el("div", {});
    const chatPanel = el("div", {});
    const frame = el("div", {},
      el("p", { class: "eyebrow" }, "Inference"),
      el("h1", {}, "Serve"),
      el("section", { class: "panel" }, sessionPanel),
      el("section", { class: "panel" }, chatPanel));

    const renderSession = () => {
      const modelSel = select(models);
      if (session) modelSel.value = session.model;
      const adapterSel = select(["", ...adapters]);
      if (preferRun) {
        const match = adapters.find(a => a.includes("/" + preferRun + "/"));
        if (match) adapterSel.value = match;
      }
      sessionPanel.replaceChildren(
        session
          ? el("table", {}, el("tbody", {},
              measureRow("Serving", session.model),
              measureRow("State", stateSpan(session.state)),
              measureRow("Endpoint", session.url + "/v1"),
              ...Object.entries(session.adapters).map(([name, key]) => measureRow("Adapter " + name, key)),
              measureRow("Idle", `${age(session.idle_for)} of ${age(session.idle_timeout)} before teardown`)))
          : el("p", { class: "empty" }, "No inference server is running. Weights stage from object storage on start."),
        el("div", { class: "row", style: "margin-top:16px" },
          el("div", {}, el("label", {}, "Model"), modelSel),
          el("div", { class: "tight actions", style: "margin-top:0" },
            el("button", { onclick: () => act(frame, start(modelSel.value), true) },
              session ? "Switch model" : "Start serving"),
            session ? el("button", { class: "danger", onclick: () => act(frame,
                api("/sessions", { method: "DELETE" }).then(() => "Stopped; GPU released."), true) }, "Stop") : null)),
        session && adapters.length ? el("div", { class: "row", style: "margin-top:16px" },
          el("div", {}, el("label", {}, "Trained adapter — loads in seconds, no merge"), adapterSel),
          el("div", { class: "tight actions", style: "margin-top:0" },
            el("button", { onclick: () => act(frame, api("/sessions/adapters", {
              method: "POST", body: JSON.stringify({ key: adapterSel.value }) })
              .then(s => { session = s; renderChat(); return "Adapter loaded."; }), true) }, "Load adapter"))) : null);
    };

    async function start(model) {
      sessionPanel.prepend(notice("ok", "Starting the server — a large base takes a minute to load…"));
      const started = await api("/sessions", { method: "POST", body: JSON.stringify({ model }) });
      session = started;
      return `Serving ${started.model}.`;
    }

    /* chat */
    const history = [];
    let attached = null;
    const renderChat = () => {
      if (!session) { chatPanel.replaceChildren(); return; }
      const names = [session.model, ...Object.keys(session.adapters)];
      const modelSel = select(names);
      modelSel.value = names[names.length - 1];
      const logDiv = el("div", { id: "chat-log" });
      const input = el("textarea", { placeholder: "Message the model — attach an image for vision models",
        onkeydown: event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(); } } });
      const fileIn = el("input", { type: "file", accept: "image/*", hidden: true, onchange: () => {
        const file = fileIn.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => { attached = reader.result; preview.textContent = file.name + " attached"; };
        reader.readAsDataURL(file);
      } });
      const preview = el("div", { id: "chat-attach-preview" });
      chatPanel.replaceChildren(
        el("p", { class: "eyebrow" }, "Chat"),
        el("div", { class: "row" }, el("div", {}, el("label", {}, "Respond as"), modelSel)),
        logDiv,
        el("form", { id: "chat-form", onsubmit: event => { event.preventDefault(); send(); } },
          input,
          el("div", { class: "tight" },
            el("button", { type: "button", class: "quiet", onclick: () => fileIn.click() }, "Image"),
            el("button", { type: "submit" }, "Send")),
          fileIn),
        preview);

      const draw = () => {
        logDiv.replaceChildren(...history.map(m => el("div", { class: "msg " + m.role },
          el("div", { class: "who" }, m.role === "user" ? "You" : m.model || "Model"),
          m.image ? el("img", { src: m.image }) : null,
          el("div", { class: "body" }, m.text))));
        logDiv.scrollTop = logDiv.scrollHeight;
      };

      async function send() {
        const text = input.value.trim();
        if (!text && !attached) return;
        const image = attached;
        attached = null; preview.textContent = ""; input.value = "";
        history.push({ role: "user", text, image });
        draw();
        const content = image
          ? [{ type: "image_url", image_url: { url: image } }, { type: "text", text }]
          : text;
        try {
          const reply = await api("/v1/chat/completions", { method: "POST", body: JSON.stringify({
            model: modelSel.value,
            messages: [...history.filter(m => !m.error).map(m => ({ role: m.role === "user" ? "user" : "assistant",
              content: m.role === "user" && m.image
                ? [{ type: "image_url", image_url: { url: m.image } }, { type: "text", text: m.text }]
                : m.text })).slice(0, -1),
              { role: "user", content }],
          }) });
          history.push({ role: "assistant", model: modelSel.value, text: reply.choices[0].message.content });
        } catch (error) {
          history.push({ role: "assistant", model: modelSel.value, text: error.message, error: true });
        }
        draw();
      }
    };

    renderSession();
    renderChat();
    return frame;
  },

  /* Agent — propose, review the diff, apply, resubmit. Never autonomous. */
  async agent(runId) {
    const status = await api("/agent");
    const instruction = el("textarea", { rows: 3, placeholder: runId
      ? "What should be fixed? The failed run's log and metrics join the context."
      : "What should be written or changed?" });
    const scriptIn = el("input", { type: "text", placeholder: "train.py (defaults to the run's script)" });
    const out = el("div", {});
    let proposal = null;

    const frame = el("div", {},
      el("p", { class: "eyebrow" }, "Agent-assisted authoring"),
      el("h1", {}, "Agent ", runId ? el("small", {}, "diagnosing " + runId) : null),
      el("section", { class: "panel" },
        status.configured
          ? el("p", { class: "muted" }, "Backend: ", el("span", { class: "mono" }, status.backend),
              ". Proposals are diffs; nothing reaches the workspace without your approval.")
          : el("p", { class: "empty" }, "No agent backend is configured — the platform works fully without one. ",
              el("span", { class: "mono" }, status.remedy || "")),
        el("label", {}, "Instruction"), instruction,
        el("label", {}, "Script"), scriptIn,
        el("div", { class: "actions" },
          el("button", { disabled: !status.configured, onclick: propose }, "Propose")),
        out));
    return frame;

    async function propose() {
      out.replaceChildren(notice("ok", "Asking the agent — this can take a minute…"));
      try {
        proposal = await api("/agent/propose", { method: "POST", body: JSON.stringify({
          instruction: instruction.value.trim(),
          script: scriptIn.value.trim() || null,
          run_id: runId || null,
        }) });
      } catch (error) { out.replaceChildren(notice("error", error.message)); return; }
      out.replaceChildren(
        el("p", { class: "eyebrow", style: "margin-top:24px" }, "Proposal"),
        el("p", {}, proposal.summary),
        ...proposal.warnings.map(w => notice("error", "⚠ " + w)),
        el("pre", { class: "log" }, proposal.diff || "(a new file)"),
        el("div", { class: "actions" },
          el("button", { onclick: apply }, "Apply to workspace"),
          el("button", { class: "quiet", onclick: propose }, "Ask again")));
    }

    async function apply() {
      try {
        const applied = await api("/agent/apply", { method: "POST", body: JSON.stringify({
          path: proposal.path, content: proposal.new_content,
          summary: proposal.summary, instruction: instruction.value.trim(), backend: proposal.backend,
        }) });
        const actions = el("div", { class: "actions" });
        if (runId) actions.append(el("button", { onclick: () => resubmit(actions) }, "Resubmit run"));
        out.append(notice("ok", `Applied and committed as ${applied.commit}.`), actions);
      } catch (error) { out.append(notice("error", error.message)); }
    }

    async function resubmit(host) {
      try {
        const failed = await api(`/runs/${runId}`);
        const record = await api("/runs", { method: "POST", body: JSON.stringify({
          script: proposal.path,
          model: failed.spec.model,
          datasets: failed.spec.datasets,
          params: failed.spec.params,
          notes: `agent revision of ${runId}`,
        }) });
        location.hash = "#run/" + record.id;
      } catch (error) { host.append(notice("error", error.message)); }
    }
  },

  /* Compare — previous traces in verdigris beneath the chosen gold. */
  async compare(anchor) {
    const records = (await api("/runs")).filter(r => r.state === "succeeded" || r.state === "failed");
    const chosen = new Set(anchor ? [anchor] : records.slice(0, 2).map(r => r.id));
    const nameIn = el("input", { type: "text", value: "loss" });
    const traceEl = traceBlock();
    const list = el("tbody", {});

    const frame = el("div", {},
      el("p", { class: "eyebrow" }, "Across runs"),
      el("h1", {}, "Compare"),
      el("section", { class: "panel" },
        el("div", { class: "row" },
          el("div", { class: "tight", style: "width:200px" }, el("label", {}, "Metric"), nameIn),
          el("div", { class: "tight actions", style: "margin-top:0" },
            el("button", { onclick: redraw }, "Read"))),
        el("div", { style: "margin-top:16px" }, traceEl)),
      el("section", { class: "panel" },
        el("p", { class: "eyebrow" }, "Runs"),
        records.length ? el("table", {}, el("thead", {}, el("tr", {},
            el("th", {}, ""), el("th", {}, "Run"), el("th", {}, "State"), el("th", {}, "Script"))), list)
          : el("p", { class: "empty" }, "Finish two runs, then read them side by side here.")));

    const renderList = () => {
      list.replaceChildren(...records.map(r => {
        const box = el("input", { type: "checkbox", onchange: () => {
          box.checked ? chosen.add(r.id) : chosen.delete(r.id); redraw();
        } });
        box.checked = chosen.has(r.id);
        return el("tr", {}, el("td", { class: "tight" }, box), el("td", {}, r.id),
          el("td", {}, stateSpan(r.state)), el("td", { class: "muted" }, r.spec.script));
      }));
    };

    async function redraw() {
      const ids = [...chosen];
      if (!ids.length) { traceEl.trace.set([]); traceEl.setLegend([]); return; }
      const query = ids.map(id => "run=" + encodeURIComponent(id)).join("&");
      const data = await api(`/metrics/compare?${query}&name=${encodeURIComponent(nameIn.value.trim() || "loss")}`);
      const series = ids.map((id, index) => ({
        label: id,
        points: (data[id] || []).map(p => ({ step: p.step, value: p.value })),
        color: index === 0 ? PALETTE.gold : COMPARE_COLORS[(index - 1) % COMPARE_COLORS.length],
        live: index === 0,
      }));
      traceEl.trace.set(series);
      traceEl.setLegend(series.map(s => ({ label: s.label, color: s.color })));
    }

    renderList();
    await redraw();
    return frame;
  },
};

function measureRow(name, value, cls = "") {
  return el("tr", {}, el("td", { class: "muted", style: "width:180px" }, name),
    el("td", { class: cls === "alarm" ? "state failed" : "" }, value.nodeType ? value : String(value)));
}

function select(options, none) {
  const sel = el("select", {}, options.map(o => el("option", { value: o }, o === "" ? "—" : o)));
  if (none && !options.length) sel.disabled = true;
  return sel;
}

/* Run an action, then re-render the current view with its outcome shown. */
async function act(host, promise, reload) {
  try {
    const message = await promise;
    if (reload !== true) route();
    else route();
    if (message) flash(message, "ok");
  } catch (error) { flash(error.message, "error"); }
}

let flashTimer;
function flash(message, kind) {
  const alerts = document.getElementById("strip-alerts");
  alerts.textContent = message;
  alerts.className = "cell " + (kind === "error" ? "alarm" : "");
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { alerts.textContent = ""; refreshStrip(); }, 8000);
}

function renderEval(host, points) {
  const evals = points.filter(p => p.event === "eval" || "cer" in p || "wer" in p);
  if (!evals.length) { host.replaceChildren(); return; }
  host.replaceChildren(el("p", { class: "eyebrow" }, "Evaluation"),
    el("table", {}, el("thead", {}, el("tr", {},
      el("th", {}, "At step"), el("th", { class: "num" }, "CER"), el("th", { class: "num" }, "WER"),
      el("th", { class: "num" }, "Samples"))),
      el("tbody", {}, evals.map(p => el("tr", {},
        el("td", {}, String(p.step ?? "—")),
        el("td", { class: "num" }, "cer" in p ? (p.cer * 100).toFixed(2) + "%" : "—"),
        el("td", { class: "num" }, "wer" in p ? (p.wer * 100).toFixed(2) + "%" : "—"),
        el("td", { class: "num muted" }, "samples" in p ? String(p.samples) : "—"))))));
}

/* ---- status strip -------------------------------------------------------- */

async function refreshStrip() {
  try {
    const health = await api("/health");
    const cacheEl = document.getElementById("strip-cache");
    const pct = Math.round(health.cache.fraction * 100);
    cacheEl.replaceChildren(
      "cache ", meterEl(health.cache.fraction), ` ${bytes(health.cache.used)} / ${bytes(health.cache.ceiling)}`);
    const gpu = document.getElementById("strip-gpu");
    gpu.textContent = health.gpu
      ? `gpu ${bytes(health.gpu.memory_used)} / ${bytes(health.gpu.memory_total)} · ${health.gpu.utilization}%`
      : "gpu —";
    const sess = document.getElementById("strip-session");
    sess.className = "cell" + (health.session ? " live" : "");
    sess.textContent = health.session ? `serving ${health.session.model}` : "idle";
    const runCell = document.getElementById("strip-run");
    runCell.className = "cell" + (health.run ? " live" : "");
    runCell.textContent = health.run ? `run ${health.run.id} ${health.run.state}` : "no active run";
    if (!flashTimer || !document.getElementById("strip-alerts").textContent) {
      const alerts = health.warnings || [];
      document.getElementById("strip-alerts").textContent = alerts.join(" · ");
    }
  } catch { /* the strip never throws */ }
}

function meterEl(fraction) {
  const meter = el("span", { class: "meter" + (fraction > 0.9 ? " hot" : "") }, el("i", {}));
  meter.firstChild.style.width = Math.min(100, Math.round(fraction * 100)) + "%";
  return meter;
}

/* ---- router -------------------------------------------------------------- */

const NAV = [
  ["storage", "Storage"], ["registry", "Registry"], ["runs", "Runs"],
  ["submit", "Submit"], ["serve", "Serve"], ["compare", "Compare"], ["agent", "Agent"],
];

function renderNav(active) {
  document.getElementById("nav").replaceChildren(...NAV.map(([id, label]) =>
    el("a", { href: "#" + id, class: id === active ? "active" : "" }, label)));
}

async function route() {
  disposeCurrent(); disposeCurrent = () => {};
  const hash = (location.hash || "#storage").slice(1);
  const [name, argument] = hash.split("/", 2);
  const view = views[name] || views.storage;
  renderNav(name === "run" ? "runs" : name);
  try {
    main.replaceChildren(await view(argument && decodeURIComponent(argument)));
  } catch (error) {
    main.replaceChildren(el("p", { class: "eyebrow" }, "Error"), notice("error", error.message));
  }
}

window.addEventListener("hashchange", route);

document.getElementById("gate-form").addEventListener("submit", event => {
  event.preventDefault();
  localStorage.setItem(TOKEN_KEY, document.getElementById("gate-token").value.trim());
  document.getElementById("gate").classList.remove("open");
  route(); refreshStrip();
});

(async function init() {
  try {
    const config = await api("/health");
    if (config.jupyter_url) {
      const link = document.getElementById("jupyter-link");
      link.href = config.jupyter_url; link.hidden = false;
    }
  } catch {}
  route();
  refreshStrip();
  setInterval(refreshStrip, 4000);
})();

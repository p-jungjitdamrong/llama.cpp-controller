/* llama-controller dashboard — no framework, no build step. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  models: [],
  selected: null,
  server: { state: "stopped" },
  info: null,
  logs: [],
  series: { cpu: [], mem: [], proc: [] },
  gpuCount: 0,
  maxPoints: 120,
};

const GPU_COLORS = ["#a371f7", "#f778ba", "#56d4dd", "#e3b341"];

/* --------------------------------------------------------------- helpers */

function bytes(n) {
  if (!n && n !== 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function duration(seconds) {
  if (!seconds) return "";
  const s = Math.floor(seconds % 60);
  const m = Math.floor((seconds / 60) % 60);
  const h = Math.floor(seconds / 3600);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || response.statusText);
  return data;
}

/* ------------------------------------------------------------ sparklines */

function drawSpark(canvas, values, color, max) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!values.length) return;

  const ceiling = max || Math.max(1, ...values) * 1.15;
  const step = width / Math.max(1, state.maxPoints - 1);
  const y = (v) => height - (Math.min(v, ceiling) / ceiling) * (height - 2) - 1;
  const x = (i) => width - (values.length - 1 - i) * step;

  ctx.strokeStyle = "#2a313c";
  ctx.lineWidth = 1;
  for (const frac of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(0, height * frac);
    ctx.lineTo(width, height * frac);
    ctx.stroke();
  }

  ctx.beginPath();
  values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.stroke();

  ctx.lineTo(x(values.length - 1), height);
  ctx.lineTo(x(0), height);
  ctx.closePath();
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, color + "44");
  gradient.addColorStop(1, color + "00");
  ctx.fillStyle = gradient;
  ctx.fill();
}

function push(series, value) {
  const arr = (state.series[series] ||= []);
  arr.push(value);
  if (arr.length > state.maxPoints) arr.shift();
}

/* GPU cards are built from whatever the backend reports: AMD, NVIDIA, Intel,
   one card or several. */
function ensureGpuCards(gpus) {
  if (state.gpuCount === gpus.length) return;
  state.gpuCount = gpus.length;
  const host = $("#gpu-cards");
  if (!gpus.length) {
    host.innerHTML = `<div class="card"><div class="card-head"><h3>GPU</h3>
      <span class="value">n/a</span></div>
      <div class="card-foot"><span class="dim tiny">no supported GPU detected</span></div></div>`;
    return;
  }
  host.innerHTML = gpus.map((gpu, i) => `
    <div class="card">
      <div class="card-head">
        <h3 title="${gpu.name || ""}">${gpu.vendor || "GPU"}${gpus.length > 1 ? ` #${i}` : ""}</h3>
        <span class="value" id="gpu-value-${i}">—</span>
      </div>
      <canvas class="spark" data-series="gpu${i}"></canvas>
      <div class="bar"><div class="bar-fill alt" id="gpu-bar-${i}"></div></div>
      <div class="card-foot"><span id="gpu-extra-${i}" class="dim tiny">—</span></div>
    </div>`).join("");
}

function renderGpus(gpus) {
  ensureGpuCards(gpus);
  gpus.forEach((gpu, i) => {
    const color = GPU_COLORS[i % GPU_COLORS.length];
    if (gpu.busy !== null && gpu.busy !== undefined) {
      push(`gpu${i}`, gpu.busy);
      $(`#gpu-value-${i}`).textContent = `${Math.round(gpu.busy)}%`;
    } else {
      $(`#gpu-value-${i}`).textContent = memPercent(gpu) !== null ? `${memPercent(gpu)}%` : "—";
    }
    drawSpark($(`canvas[data-series="gpu${i}"]`), state.series[`gpu${i}`] || [], color, 100);

    const pct = memPercent(gpu);
    $(`#gpu-bar-${i}`).style.width = `${pct ?? 0}%`;
    const bits = [];
    if (gpu.mem_total) {
      bits.push(`${gpu.mem_label || "mem"} ${bytes(gpu.mem_used)}/${bytes(gpu.mem_total)}`);
    }
    if (gpu.extra && gpu.extra.gtt_total && gpu.mem_label !== "GTT") {
      bits.push(`GTT ${bytes(gpu.extra.gtt_used)}`);
    }
    if (gpu.temp) bits.push(`${gpu.temp}°C`);
    if (gpu.power) bits.push(`${gpu.power} W`);
    if (gpu.clock_mhz) bits.push(`${Math.round(gpu.clock_mhz)} MHz`);
    if (gpu.busy === null || gpu.busy === undefined) bits.push("busy% not exposed");
    $(`#gpu-extra-${i}`).textContent = bits.join(" · ") || gpu.name || "—";
  });
}

function memPercent(gpu) {
  if (!gpu.mem_total || gpu.mem_used === null || gpu.mem_used === undefined) return null;
  return Math.round((gpu.mem_used / gpu.mem_total) * 100);
}

/* --------------------------------------------------------------- metrics */

function renderMetrics(sample) {
  const { cpu, mem, process: proc } = sample;

  push("cpu", cpu.percent);
  $("#cpu-value").textContent = `${cpu.percent.toFixed(0)}%`;
  drawSpark($('canvas[data-series="cpu"]'), state.series.cpu, "#58a6ff", 100);
  const cores = $("#cpu-cores");
  if (cores.children.length !== cpu.per_core.length) {
    cores.innerHTML = cpu.per_core.map(() => '<div class="core"></div>').join("");
  }
  cpu.per_core.forEach((v, i) => {
    cores.children[i].style.height = `${Math.max(2, v)}%`;
    cores.children[i].style.opacity = 0.35 + (v / 100) * 0.65;
  });
  const load = sample.load.map((v) => v.toFixed(2)).join(" ");
  $("#cpu-extra").textContent =
    `load ${load}` + (sample.cpu_temp ? ` · ${sample.cpu_temp}°C` : "");

  push("mem", mem.percent);
  $("#mem-value").textContent = `${mem.percent.toFixed(0)}%`;
  $("#mem-bar").style.width = `${mem.percent}%`;
  drawSpark($('canvas[data-series="mem"]'), state.series.mem, "#3fb950", 100);
  $("#mem-extra").textContent =
    `${bytes(mem.used)} / ${bytes(mem.total)} · ${bytes(mem.available)} free` +
    (mem.swap_total ? ` · swap ${bytes(mem.swap_used)}` : "");

  renderGpus(sample.gpus || []);

  if (proc) {
    push("proc", proc.cpu_percent);
    $("#proc-value").textContent = `${proc.cpu_percent.toFixed(0)}%`;
    drawSpark($('canvas[data-series="proc"]'), state.series.proc, "#d29922", null);
    $("#proc-extra").textContent =
      `pid ${proc.pid} · RSS ${bytes(proc.rss)} · ${proc.threads} threads · up ${duration(proc.elapsed)}`;
  } else {
    $("#proc-value").textContent = "—";
    $("#proc-extra").textContent = "not running";
    state.series.proc = [];
    drawSpark($('canvas[data-series="proc"]'), [], "#d29922", null);
  }
}

/* ---------------------------------------------------------------- status */

function renderServer(server) {
  state.server = server;
  const pill = $("#state-pill");
  pill.textContent = server.state;
  pill.dataset.state = server.state;

  $("#running-model").textContent = server.model_name || "no model loaded";
  $("#uptime").textContent = server.pid ? `up ${duration(server.uptime)}` : "";

  // what someone on another machine should point their client at
  const link = $("#endpoint-link");
  if (server.pid && server.bind_port) {
    const host = ["0.0.0.0", "::", ""].includes(server.bind_host)
      ? location.hostname : server.bind_host;
    link.href = `http://${host}:${server.bind_port}`;
    link.textContent = `${host}:${server.bind_port}`;
    link.title = server.bind_host === "0.0.0.0"
      ? "reachable from other machines" : "bound to " + server.bind_host;
    link.hidden = false;
  } else {
    link.hidden = true;
  }
  $("#btn-stop").disabled = !server.pid;
  $("#btn-restart").disabled = !server.model_path;
  $("#chat-send").disabled = server.state !== "ready";
  $("#d-argv").textContent = (server.argv || []).join(" ") || "—";

  const rows = [];
  if (server.load_seconds) rows.push(["load time", `${server.load_seconds}s`]);
  if (server.url) rows.push(["endpoint", server.url]);
  const stats = server.stats || {};
  if (stats.prompt) {
    rows.push(["prompt eval",
      `${stats.prompt.tokens} tok @ ${(stats.prompt.tokens_per_second || 0).toFixed(1)} tok/s`]);
  }
  if (stats.generation) {
    rows.push(["generation",
      `${stats.generation.tokens} tok @ ${(stats.generation.tokens_per_second || 0).toFixed(1)} tok/s`]);
  }
  const meta = server.model_meta || {};
  if (meta.n_ctx) rows.push(["n_ctx", meta.n_ctx.toLocaleString()]);
  if (meta.slots) rows.push(["slots", meta.slots]);
  if (meta.build) rows.push(["build", meta.build]);
  if (server.exit_code !== null && server.exit_code !== undefined) {
    rows.push(["exit code", server.exit_code]);
  }
  if (server.last_error) rows.push(["last error", server.last_error]);
  $("#d-stats").innerHTML = rows.length
    ? rows.map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`).join("")
    : "—";

  $$(".model").forEach((el) => el.classList.toggle("running", el.dataset.path === server.model_path));
}

/* ---------------------------------------------------------------- models */

function modelRow(model) {
  const meta = [];
  if (model.quant || model.quant_guess) meta.push(`<span class="tag">${model.quant || model.quant_guess}</span>`);
  if (model.size_label || model.size_guess) meta.push(model.size_label || model.size_guess);
  if (model.architecture) meta.push(model.architecture);
  meta.push(bytes(model.size));
  if (model.n_ctx_train) meta.push(`ctx ${model.n_ctx_train.toLocaleString()}`);
  if (model.n_layer) meta.push(`${model.n_layer}L`);
  return `<div class="model" data-path="${model.path}">
      <div class="model-name">${model.name}</div>
      <div class="model-meta">${meta.join(" ")}</div>
    </div>`;
}

function renderModels() {
  const filter = $("#model-filter").value.toLowerCase();
  const visible = state.models.filter((m) => m.name.toLowerCase().includes(filter));
  const list = $("#model-list");
  list.innerHTML = visible.length
    ? visible.map(modelRow).join("")
    : '<p class="empty">no .gguf files found</p>';

  list.querySelectorAll(".model").forEach((el) => {
    el.classList.toggle("selected", el.dataset.path === state.selected);
    el.classList.toggle("running", el.dataset.path === state.server.model_path);
    el.onclick = () => selectModel(el.dataset.path);
  });
}

function selectModel(path) {
  state.selected = path;
  const model = state.models.find((m) => m.path === path);
  if (model && model.params) setParams(model.params);
  $("#btn-start").disabled = false;
  $("#btn-start").textContent = state.server.pid ? "Switch to this model" : "Start model";
  renderModels();
  updatePreview();
}

async function loadModels() {
  $("#model-list").innerHTML = '<p class="empty">scanning…</p>';
  try {
    const data = await api("/api/models");
    state.models = data.models;
    $("#model-dirs").textContent = `scanned: ${data.dirs.join(", ")}`;
    renderModels();
  } catch (err) {
    $("#model-list").innerHTML = `<p class="empty">scan failed: ${err.message}</p>`;
  }
}

/* ---------------------------------------------------------------- params */

function getParams() {
  return {
    host: $("#p-host").value.trim() || "0.0.0.0",
    port: +$("#p-port").value,
    n_gpu_layers: +$("#p-ngl").value,
    ctx_size: +$("#p-ctx").value,
    threads: +$("#p-threads").value,
    parallel: +$("#p-parallel").value,
    batch_size: +$("#p-batch").value,
    flash_attn: $("#p-fa").value,
    mlock: $("#p-mlock").checked,
    no_mmap: $("#p-nommap").checked,
    jinja: $("#p-jinja").checked,
    extra_args: $("#p-extra").value,
  };
}

function setParams(p) {
  $("#p-host").value = p.host ?? "0.0.0.0";
  $("#p-port").value = p.port ?? 8090;
  $("#p-ngl").value = p.n_gpu_layers;
  $("#p-ctx").value = p.ctx_size;
  $("#p-threads").value = p.threads;
  $("#p-parallel").value = p.parallel;
  $("#p-batch").value = p.batch_size;
  $("#p-fa").value = p.flash_attn;
  $("#p-mlock").checked = p.mlock;
  $("#p-nommap").checked = p.no_mmap;
  $("#p-jinja").checked = p.jinja;
  $("#p-extra").value = p.extra_args || "";
}

function updatePreview() {
  if (!state.selected || !state.info) { $("#cmd-preview").textContent = ""; return; }
  const p = getParams();
  const cfg = state.info.config;
  const argv = [cfg.llama_server_bin, "-m", state.selected,
    "--host", p.host, "--port", p.port, "--metrics",
    "-ngl", p.n_gpu_layers, "-c", p.ctx_size, "-np", p.parallel];
  if (p.threads) argv.push("-t", p.threads);
  if (p.batch_size) argv.push("-b", p.batch_size);
  if (p.flash_attn) argv.push("-fa", p.flash_attn);
  if (p.mlock) argv.push("--mlock");
  if (p.no_mmap) argv.push("--no-mmap");
  argv.push(p.jinja ? "--jinja" : "--no-jinja");
  if (p.extra_args.trim()) argv.push(p.extra_args.trim());
  $("#cmd-preview").textContent = argv.join(" ");
}

/* ------------------------------------------------------------------ logs */

function logLine(record) {
  const time = new Date(record.ts * 1000).toLocaleTimeString();
  let cls = record.stream === "controller" ? "controller" : "";
  if (/error|failed|abort/i.test(record.line)) cls = "err";
  else if (/warn/i.test(record.line)) cls = "warn";
  const escaped = record.line.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  return `<span class="l ${cls}"><span class="ts">${time}</span>${escaped}</span>`;
}

function appendLog(record) {
  state.logs.push(record);
  if (state.logs.length > 5000) state.logs.shift();
  const filter = $("#log-filter").value.toLowerCase();
  if (filter && !record.line.toLowerCase().includes(filter)) return;
  const view = $("#log-view");
  const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 40;
  view.insertAdjacentHTML("beforeend", logLine(record));
  if ($("#log-autoscroll").checked && atBottom) view.scrollTop = view.scrollHeight;
}

function renderLogs() {
  const filter = $("#log-filter").value.toLowerCase();
  const view = $("#log-view");
  view.innerHTML = state.logs
    .filter((r) => !filter || r.line.toLowerCase().includes(filter))
    .map(logLine)
    .join("");
  view.scrollTop = view.scrollHeight;
}

/* ------------------------------------------------------------- websocket */

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => $("#ws-dot").classList.add("on");
  ws.onclose = () => {
    $("#ws-dot").classList.remove("on");
    setTimeout(connect, 2000);
  };
  ws.onmessage = (event) => {
    const { type, data } = JSON.parse(event.data);
    if (type === "metrics") {
      renderMetrics(data);
      renderServer(data.server);
    } else if (type === "log") {
      appendLog(data);
    } else if (type === "hello") {
      state.logs = data.logs || [];
      renderLogs();
      renderServer(data.server);
      data.history.forEach((s) => {
        push("cpu", s.cpu.percent);
        push("mem", s.mem.percent);
        (s.gpus || []).forEach((g, i) => {
          if (g.busy !== null && g.busy !== undefined) push(`gpu${i}`, g.busy);
        });
        if (s.process) push("proc", s.process.cpu_percent);
      });
    }
  };
}

/* ------------------------------------------------------------------ chat */

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  const log = $("#chat-log");
  log.querySelector(".empty")?.remove();
  log.insertAdjacentHTML("beforeend", `<div class="msg user"></div>`);
  log.lastElementChild.textContent = text;
  log.insertAdjacentHTML("beforeend", `<div class="msg assistant"></div>`);
  const bubble = log.lastElementChild;
  log.scrollTop = log.scrollHeight;

  const started = performance.now();
  let tokens = 0;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: text }], max_tokens: 512 }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n");
      buffer = parts.pop();
      for (const line of parts) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") continue;
        try {
          const chunk = JSON.parse(payload);
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) { bubble.textContent += delta; tokens++; log.scrollTop = log.scrollHeight; }
        } catch { /* partial frame */ }
      }
    }
    const seconds = (performance.now() - started) / 1000;
    $("#chat-stats").textContent =
      `${tokens} chunks in ${seconds.toFixed(1)}s (~${(tokens / seconds).toFixed(1)}/s)`;
  } catch (err) {
    bubble.classList.add("err");
    bubble.textContent = `error: ${err.message}`;
  }
}

/* ------------------------------------------------------------------ init */

async function init() {
  state.info = await api("/api/info");
  $("#hostname").textContent = state.info.system.hostname;
  $("#cpu-model").textContent =
    `${state.info.system.cpu_model} · ${state.info.system.cpu_count} threads · ${bytes(state.info.system.mem_total)}`;
  setParams(state.info.config.defaults);
  $("#d-system").innerHTML = Object.entries(state.info.system)
    .map(([k, v]) => {
      const text = k === "gpus"
        ? (v.length ? v.map((g) => `${g.vendor} ${g.name} (${g.driver})`).join(", ") : "none")
        : (k === "mem_total" ? bytes(v) : v);
      return `<span class="k">${k}</span><span class="v">${text}</span>`;
    }).join("");
  const external = state.info.external_servers;
  $("#d-external").innerHTML = external.length
    ? external.map((p) => `<span class="k">pid ${p.pid}</span><span class="v">${p.cmdline}</span>`).join("")
    : '<span class="k">none</span><span class="v">—</span>';

  await loadModels();
  if (state.info.config.last_model &&
      state.models.some((m) => m.path === state.info.config.last_model)) {
    selectModel(state.info.config.last_model);
  }
  connect();
}

$("#btn-start").onclick = async () => {
  if (!state.selected) return;
  const button = $("#btn-start");
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    await api("/api/server/start", {
      method: "POST",
      body: JSON.stringify({ model_path: state.selected, params: getParams() }),
    });
    document.querySelector('.tab[data-tab="logs"]').click();
  } catch (err) {
    alert(`Failed to start: ${err.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Start model";
  }
};

$("#btn-stop").onclick = () => api("/api/server/stop", { method: "POST" });
$("#btn-restart").onclick = () => api("/api/server/restart", { method: "POST" });
$("#btn-rescan").onclick = loadModels;
$("#model-filter").oninput = renderModels;
$("#log-filter").oninput = renderLogs;
$("#btn-clear-log").onclick = () => { state.logs = []; renderLogs(); };
$("#chat-form").onsubmit = sendChat;
$("#chat-input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); }
};
$$(".params input, .params select").forEach((el) => el.addEventListener("input", updatePreview));
$$(".tab").forEach((tab) => {
  tab.onclick = () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab.dataset.tab}`));
  };
});
window.addEventListener("resize", () => {
  const colors = { cpu: "#58a6ff", mem: "#3fb950", proc: "#d29922" };
  $$("canvas.spark").forEach((canvas) => {
    const key = canvas.dataset.series;
    const gpuIndex = key.startsWith("gpu") ? +key.slice(3) : null;
    const color = gpuIndex === null ? colors[key] : GPU_COLORS[gpuIndex % GPU_COLORS.length];
    drawSpark(canvas, state.series[key] || [], color, key === "proc" ? null : 100);
  });
});

init().catch((err) => alert(`init failed: ${err.message}`));

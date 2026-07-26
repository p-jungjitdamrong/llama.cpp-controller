/* llama-controller dashboard — no framework, no build step. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  models: [],
  selected: null,
  servers: [],          // one entry per llama-server this controller runs
  info: null,
  logs: [],
  mode: "single",       // single | router
  expandedRouters: new Set(),
  serverSignature: null,
  series: { cpu: [], mem: [], proc: [] },
  gpuCount: 0,
  maxPoints: 120,
};

const GPU_COLORS = ["#a371f7", "#f778ba", "#56d4dd", "#e3b341"];
const LIVE = ["starting", "ready", "stopping"];

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

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
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
  if (!canvas) return;
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

/* -------------------------------------------------------------- GPU cards */

function memPercent(gpu) {
  if (!gpu.mem_total || gpu.mem_used === null || gpu.mem_used === undefined) return null;
  return Math.round((gpu.mem_used / gpu.mem_total) * 100);
}

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
        <h3 title="${escapeHtml(gpu.name || "")}">${escapeHtml(gpu.vendor || "GPU")}${gpus.length > 1 ? ` #${i}` : ""}</h3>
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
    const pct = memPercent(gpu);
    if (gpu.busy !== null && gpu.busy !== undefined) {
      push(`gpu${i}`, gpu.busy);
      $(`#gpu-value-${i}`).textContent = `${Math.round(gpu.busy)}%`;
    } else {
      $(`#gpu-value-${i}`).textContent = pct !== null ? `${pct}%` : "—";
    }
    drawSpark($(`canvas[data-series="gpu${i}"]`), state.series[`gpu${i}`] || [], color, 100);
    $(`#gpu-bar-${i}`).style.width = `${pct ?? 0}%`;

    const bits = [];
    if (gpu.mem_total) bits.push(`${gpu.mem_label || "mem"} ${bytes(gpu.mem_used)}/${bytes(gpu.mem_total)}`);
    if (gpu.extra && gpu.extra.gtt_total && gpu.mem_label !== "GTT") bits.push(`GTT ${bytes(gpu.extra.gtt_used)}`);
    if (gpu.temp) bits.push(`${gpu.temp}°C`);
    if (gpu.power) bits.push(`${gpu.power} W`);
    if (gpu.clock_mhz) bits.push(`${Math.round(gpu.clock_mhz)} MHz`);
    if (gpu.busy === null || gpu.busy === undefined) bits.push("busy% not exposed");
    $(`#gpu-extra-${i}`).textContent = bits.join(" · ") || gpu.name || "—";
  });
}

/* --------------------------------------------------------------- metrics */

function renderMetrics(sample) {
  const { cpu, mem } = sample;

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
  $("#cpu-extra").textContent =
    `load ${sample.load.map((v) => v.toFixed(2)).join(" ")}` +
    (sample.cpu_temp ? ` · ${sample.cpu_temp}°C` : "");

  push("mem", mem.percent);
  $("#mem-value").textContent = `${mem.percent.toFixed(0)}%`;
  $("#mem-bar").style.width = `${mem.percent}%`;
  drawSpark($('canvas[data-series="mem"]'), state.series.mem, "#3fb950", 100);
  $("#mem-extra").textContent =
    `${bytes(mem.used)} / ${bytes(mem.total)} · ${bytes(mem.available)} free` +
    (mem.swap_total ? ` · swap ${bytes(mem.swap_used)}` : "");

  renderGpus(sample.gpus || []);

  const totals = sample.process_total || { count: 0 };
  if (totals.count) {
    push("proc", totals.cpu_percent);
    $("#proc-value").textContent = `${totals.cpu_percent.toFixed(0)}%`;
    drawSpark($('canvas[data-series="proc"]'), state.series.proc, "#d29922", null);
    $("#proc-extra").textContent =
      `${totals.count} process${totals.count === 1 ? "" : "es"} · RSS ${bytes(totals.rss)} · ${totals.threads} threads`;
  } else {
    $("#proc-value").textContent = "—";
    $("#proc-extra").textContent = "nothing running";
    state.series.proc = [];
    drawSpark($('canvas[data-series="proc"]'), [], "#d29922", null);
  }
}

/* --------------------------------------------------------------- servers */

function serverMeta(server) {
  const meta = [];
  if (server.pid) meta.push(`pid ${server.pid}`);
  if (server.load_seconds) meta.push(`loaded in ${server.load_seconds}s`);
  if (server.uptime) meta.push(`up ${duration(server.uptime)}`);
  if (server.process) meta.push(`${server.process.cpu_percent.toFixed(0)}% CPU · ${bytes(server.process.rss)}`);
  const gen = server.stats && server.stats.generation;
  if (gen && gen.tokens_per_second) meta.push(`${gen.tokens_per_second.toFixed(1)} tok/s`);
  if (server.model_meta && server.model_meta.n_ctx) meta.push(`ctx ${server.model_meta.n_ctx.toLocaleString()}`);
  return meta.join(" · ");
}

function serverCard(server) {
  const host = ["0.0.0.0", "::", ""].includes(server.bind_host) ? location.hostname : server.bind_host;
  const url = `http://${host}:${server.bind_port}`;
  const stopped = !server.pid;

  return `<div class="server" data-id="${server.id}">
      <div class="server-head">
        <span class="pill" data-state="${server.state}">${server.state}</span>
        <span class="server-name">${escapeHtml(server.model_name)}</span>
        <a class="endpoint" href="${url}" target="_blank" rel="noopener">${escapeHtml(url)}</a>
        <span class="server-actions">
          <button class="btn small" data-action="restart" data-id="${server.id}">Restart</button>
          <button class="btn small ${stopped ? "" : "danger"}" data-action="${stopped ? "remove" : "stop"}"
                  data-id="${server.id}">${stopped ? "Remove" : "Stop"}</button>
        </span>
      </div>
      <div class="server-meta dim tiny">${serverMeta(server) || "&nbsp;"}</div>
      ${server.last_error ? `<div class="server-error tiny">${escapeHtml(server.last_error)}</div>` : ""}
      ${server.is_router ? routerModels(server) : ""}
    </div>`;
}

function routerModels(server) {
  if (!server.router_models.length) {
    return '<div class="router-models dim tiny">router has no models yet</div>';
  }
  const open = state.expandedRouters.has(server.id);
  const loadedModels = server.router_models.filter((m) => m.status === "loaded");
  const rows = server.router_models.map((m) => {
    const loaded = m.status === "loaded";
    return `<div class="router-row">
        <span class="dot-state ${loaded ? "on" : ""}"></span>
        <span class="router-id" title="${escapeHtml(m.id)}">${escapeHtml(m.id)}</span>
        <span class="dim tiny">${escapeHtml(m.source || "")}</span>
        <button class="btn small ghost" data-router="${loaded ? "unload" : "load"}"
                data-id="${server.id}" data-model="${escapeHtml(m.id)}">${loaded ? "Unload" : "Load"}</button>
      </div>`;
  }).join("");
  // collapsed by default: the summary already says what is resident
  const resident = loadedModels.length
    ? loadedModels.map((m) => escapeHtml(m.id.split("/").pop())).join(", ")
    : "none loaded";
  return `<div class="router-models">
      <div class="router-summary" data-toggle="${server.id}">
        <span class="chevron ${open ? "open" : ""}">▸</span>
        <span class="dim tiny">${server.router_models.length} models · ${loadedModels.length}/${server.params.models_max} loaded</span>
        <span class="dim tiny resident">${resident}</span>
      </div>
      <div class="router-rows ${open ? "" : "hidden"}">${rows}</div>
    </div>`;
}

function renderServers(servers) {
  state.servers = servers;
  const live = servers.filter((s) => LIVE.includes(s.state));
  const ready = servers.filter((s) => s.state === "ready");

  const pill = $("#summary-pill");
  pill.textContent = live.length ? `${live.length} running` : "no servers";
  pill.dataset.state = ready.length ? "ready" : live.length ? "starting" : "stopped";
  $("#summary-detail").textContent = ready.length
    ? ready.map((s) => `${s.model_name} :${s.bind_port}`).join(" · ")
    : "";
  $("#btn-stop-all").disabled = !live.length;

  // Rebuild only when something structural changed; otherwise just refresh the
  // volatile numbers, so the list does not flicker or lose scroll every second.
  const signature = JSON.stringify(servers.map((s) => [
    s.id, s.state, s.pid, s.bind_port, s.model_name, s.last_error,
    state.expandedRouters.has(s.id),
    s.router_models.map((m) => m.id + m.status),
  ]));
  const list = $("#server-list");
  if (signature === state.serverSignature) {
    for (const server of servers) {
      const meta = list.querySelector(`.server[data-id="${server.id}"] .server-meta`);
      if (meta) meta.textContent = serverMeta(server);
    }
    syncInstanceSelects(servers);
    return;
  }
  state.serverSignature = signature;

  list.innerHTML = servers.length
    ? servers.map(serverCard).join("")
    : '<p class="empty">no servers started yet</p>';

  list.querySelectorAll(".router-summary").forEach((row) => {
    row.onclick = () => {
      const id = +row.dataset.toggle;
      if (state.expandedRouters.has(id)) state.expandedRouters.delete(id);
      else state.expandedRouters.add(id);
      state.serverSignature = null;
      renderServers(state.servers);
    };
  });

  list.querySelectorAll("button[data-action]").forEach((button) => {
    button.onclick = async () => {
      const { action, id } = button.dataset;
      button.disabled = true;
      try {
        await api(`/api/server/${action}`, { method: "POST", body: JSON.stringify({ id: +id }) });
      } catch (err) {
        alert(err.message);
        button.disabled = false;
      }
    };
  });
  list.querySelectorAll("button[data-router]").forEach((button) => {
    button.onclick = async () => {
      const label = button.textContent;
      button.disabled = true;
      button.textContent = button.dataset.router === "load" ? "loading…" : "unloading…";
      try {
        await api(`/api/router/${button.dataset.router}`, {
          method: "POST",
          body: JSON.stringify({ id: +button.dataset.id, model: button.dataset.model }),
        });
      } catch (err) {
        alert(err.message);
        button.textContent = label;
        button.disabled = false;
      }
    };
  });

  syncInstanceSelects(servers);
  $$(".model").forEach((el) => {
    const running = servers.find((s) => s.model_path === el.dataset.path && s.pid);
    el.classList.toggle("running", Boolean(running));
  });
}

function renderAutostart(entries) {
  state.autostart = entries;
  const hint = $("#servers-hint");
  if (!entries.length) {
    hint.textContent = "nothing starts automatically";
    hint.title = "";
    return;
  }
  const names = entries.map((e) => {
    const params = e.params || {};
    return params.mode === "router"
      ? `router:${params.models_dir.split("/").pop()}`
      : e.model_path.split("/").pop();
  });
  hint.textContent = `on startup: ${names.length} server${names.length === 1 ? "" : "s"}`;
  hint.title = names.join("\n");
}

function syncInstanceSelects(servers) {
  const options = servers
    .filter((s) => LIVE.includes(s.state))
    .map((s) => `<option value="${s.id}">${escapeHtml(s.model_name)} :${s.bind_port}</option>`)
    .join("");

  const logSelect = $("#log-instance");
  const keepLog = logSelect.value;
  logSelect.innerHTML = `<option value="">all servers</option>${options}`;
  logSelect.value = keepLog;

  const chatSelect = $("#chat-instance");
  const keepChat = chatSelect.value;
  chatSelect.innerHTML = options || '<option value="">no server running</option>';
  if ([...chatSelect.options].some((o) => o.value === keepChat)) chatSelect.value = keepChat;
  updateChatModels();
}

function updateChatModels() {
  const server = state.servers.find((s) => String(s.id) === $("#chat-instance").value);
  const select = $("#chat-model");
  if (server && server.is_router && server.router_models.length) {
    const keep = select.value;
    select.innerHTML = server.router_models
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.id)}${m.status === "loaded" ? " ✓" : ""}</option>`)
      .join("");
    if ([...select.options].some((o) => o.value === keep)) select.value = keep;
    select.hidden = false;
  } else {
    select.hidden = true;
    select.innerHTML = "";
  }
  $("#chat-send").disabled = !(server && server.state === "ready");
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
  const repo = model.repo ? `<div class="model-repo">${escapeHtml(model.repo)}</div>` : "";
  return `<div class="model" data-path="${escapeHtml(model.path)}" title="${escapeHtml(model.path)}">
      <div class="model-name">${escapeHtml(model.name)}</div>
      ${repo}
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
    el.classList.toggle("running",
      state.servers.some((s) => s.model_path === el.dataset.path && s.pid));
    el.onclick = () => selectModel(el.dataset.path);
  });
}

function selectModel(path) {
  state.selected = path;
  const model = state.models.find((m) => m.path === path);
  if (model && model.params) setParams(model.params);
  setMode("single");
  renderModels();
  updateStartButton();
  updatePreview();
}

async function loadModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models;
    $("#model-count").textContent = data.models.length || "";
    renderScanDetails(data);
    renderModels();
    renderManage(data);
  } catch (err) {
    $("#model-list").innerHTML = `<p class="empty">scan failed: ${escapeHtml(err.message)}</p>`;
  }
}

/* Folder list and skipped files, kept behind the ⓘ button so the card stays short. */
function renderScanDetails(data) {
  const byReason = new Map();
  for (const item of data.skipped || []) {
    byReason.set(item.reason, (byReason.get(item.reason) || 0) + 1);
  }
  const skipped = [...byReason].map(([reason, n]) => `<div>${n} × ${escapeHtml(reason)}</div>`);
  $("#model-dirs").innerHTML = `
    <div class="drawer-title">scanned</div>
    ${data.dirs.map((d) => `<div>${escapeHtml(d)}</div>`).join("")}
    ${skipped.length ? `<div class="drawer-title">skipped</div>${skipped.join("")}` : ""}`;
  $("#model-dirs").title = (data.skipped || [])
    .map((s) => `${s.name} — ${s.reason}`).join("\n");
}

/* ------------------------------------------------------------- management */

function manageRow(model) {
  const running = model.running_port;
  const badges = [];
  if (running) badges.push(`<span class="badge on">:${running}</span>`);
  if (model.autostart) badges.push('<span class="badge">startup</span>');
  const source = model.repo
    ? `<span class="badge dim-badge" title="${escapeHtml(model.dir)}">${escapeHtml(model.repo)}</span>`
    : `<span class="mpath dim tiny">${escapeHtml(model.dir)}</span>`;
  return `<tr data-path="${escapeHtml(model.path)}">
      <td><div class="mname">${escapeHtml(model.name)}</div>
          <div>${source}</div></td>
      <td>${escapeHtml(model.quant || model.quant_guess || "—")}</td>
      <td>${escapeHtml(model.architecture || "—")}</td>
      <td class="num">${bytes(model.size)}</td>
      <td class="num">${model.n_ctx_train ? model.n_ctx_train.toLocaleString() : "—"}</td>
      <td>${badges.join(" ") || '<span class="dim tiny">idle</span>'}</td>
      <td class="right nowrap">
        <button class="btn small" data-mstart="${escapeHtml(model.path)}" ${running ? "disabled" : ""}>Start</button>
        <button class="btn small danger" data-mdelete="${escapeHtml(model.path)}"
                data-name="${escapeHtml(model.name)}" data-size="${model.size}">Delete</button>
      </td>
    </tr>`;
}

function renderManage(data) {
  const models = data.models;
  const rows = $("#manage-rows");
  rows.innerHTML = models.length
    ? models.map(manageRow).join("")
    : '<tr><td colspan="7" class="empty">no models found</td></tr>';

  const totalSize = models.reduce((sum, m) => sum + m.size, 0);
  const disks = (data.dir_info || []).filter((d) => d.exists && d.free);
  const seen = new Set();
  const diskText = disks.filter((d) => !seen.has(d.free) && seen.add(d.free))
    .map((d) => `${bytes(d.free)} free`).join(" · ");
  $("#storage-info").innerHTML =
    `<strong>${models.length}</strong> models · <strong>${bytes(totalSize)}</strong> on disk` +
    (diskText ? ` · ${diskText}` : "") +
    `<div class="dim tiny">downloads go to ${escapeHtml(data.download_dir || "")}</div>`;

  rows.querySelectorAll("button[data-mstart]").forEach((button) => {
    button.onclick = () => { selectModel(button.dataset.mstart); $("#btn-start").click(); };
  });
  rows.querySelectorAll("button[data-mdelete]").forEach((button) => {
    button.onclick = async () => {
      const { name, size } = button.dataset;
      if (!confirm(`Delete ${name} (${bytes(+size)})?\n\nThe file is removed from disk — this cannot be undone.`)) return;
      button.disabled = true;
      try {
        await api("/api/models/delete", {
          method: "POST",
          body: JSON.stringify({ path: button.dataset.mdelete, confirm: true }),
        });
        await loadModels();
      } catch (err) {
        alert(err.message);
        button.disabled = false;
      }
    };
  });

  const skipped = data.skipped || [];
  $("#skipped-list").innerHTML = skipped.length
    ? skipped.map((s) => `<div>${escapeHtml(s.name)} <span class="dim">— ${escapeHtml(s.reason)}</span></div>`).join("")
    : '<div class="dim">nothing skipped</div>';
}

async function loadExternal() {
  const data = await api("/api/processes");
  const host = $("#external-list");
  host.innerHTML = data.external.length
    ? data.external.map((p) => `<div class="server">
        <div class="server-head">
          <span class="pill" data-state="starting">external</span>
          <span class="server-name">pid ${p.pid}${p.port ? ` · port ${p.port}` : ""}</span>
          <span class="server-actions">
            <button class="btn small danger" data-kill="${p.pid}">Kill</button>
          </span>
        </div>
        <div class="server-meta dim tiny">${escapeHtml(p.cmdline)}</div>
      </div>`).join("")
    : '<p class="empty">none — every llama-server on this box is managed here</p>';

  host.querySelectorAll("button[data-kill]").forEach((button) => {
    button.onclick = async () => {
      const pid = button.dataset.kill;
      if (!confirm(`Kill llama-server pid ${pid}?\n\nIt was not started by this controller.`)) return;
      button.disabled = true;
      button.textContent = "killing…";
      try {
        await api("/api/processes/kill", { method: "POST", body: JSON.stringify({ pid: +pid }) });
        await loadExternal();
      } catch (err) {
        alert(err.message);
        button.textContent = "Kill";
        button.disabled = false;
      }
    };
  });
}

/* ---------------------------------------------------------------- params */

function getParams() {
  return {
    mode: state.mode,
    host: $("#p-host").value.trim() || "0.0.0.0",
    port: +$("#p-port").value,
    models_dir: $("#p-models-dir").value.trim(),
    models_max: +$("#p-models-max").value,
    models_autoload: $("#p-autoload").checked,
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
  if (p.models_dir) $("#p-models-dir").value = p.models_dir;
  $("#p-models-max").value = p.models_max ?? 2;
  $("#p-autoload").checked = p.models_autoload !== false;
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

function setMode(mode) {
  state.mode = mode;
  $$(".mode").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $$(".single-only").forEach((el) => el.classList.toggle("hidden", mode !== "single"));
  $$(".router-only").forEach((el) => el.classList.toggle("hidden", mode !== "router"));
  $("#mode-help").textContent = mode === "router"
    ? "One llama-server hosts a whole directory and loads models on demand. The flags below are written to a preset the router applies to every model it loads."
    : "Starts one llama-server per model — run as many as memory allows.";
  updateStartButton();
  updatePreview();
}

function updateStartButton() {
  const button = $("#btn-start");
  if (state.mode === "router") {
    button.disabled = !$("#p-models-dir").value.trim();
    button.textContent = "Start router";
  } else {
    button.disabled = !state.selected;
    button.textContent = state.selected ? "Start model" : "Select a model to start";
  }
}

function updatePreview() {
  if (!state.info) return;
  const p = getParams();
  const bin = state.info.config.llama_server_bin;
  let argv;
  if (p.mode === "router") {
    if (!p.models_dir) { $("#cmd-preview").textContent = ""; return; }
    argv = [bin, "--host", p.host, "--port", p.port, "--metrics",
      "--models-dir", p.models_dir, "--models-max", p.models_max,
      p.models_autoload ? "--models-autoload" : "--no-models-autoload"];
    if (!p.extra_args.includes("--models-preset")) {
      argv.push("--models-preset", `router-preset-${p.port}.ini`);
    }
  } else {
    if (!state.selected) { $("#cmd-preview").textContent = ""; return; }
    argv = [bin, "--host", p.host, "--port", p.port, "--metrics",
      "-m", state.selected, "-ngl", p.n_gpu_layers, "-c", p.ctx_size, "-np", p.parallel];
    if (p.threads) argv.push("-t", p.threads);
    if (p.batch_size) argv.push("-b", p.batch_size);
    if (p.flash_attn) argv.push("-fa", p.flash_attn);
    if (p.mlock) argv.push("--mlock");
    if (p.no_mmap) argv.push("--no-mmap");
    argv.push(p.jinja ? "--jinja" : "--no-jinja");
  }
  if (p.extra_args.trim()) argv.push(p.extra_args.trim());
  $("#cmd-preview").textContent = argv.join(" ");
}

/* ------------------------------------------------------------------ logs */

function logLine(record) {
  const time = new Date(record.ts * 1000).toLocaleTimeString();
  let cls = record.stream === "controller" ? "controller" : "";
  if (/error|failed|abort/i.test(record.line)) cls = "err";
  else if (/warn/i.test(record.line)) cls = "warn";
  const tag = record.model ? `<span class="ltag">${escapeHtml(record.model.slice(0, 22))}</span>` : "";
  return `<span class="l ${cls}"><span class="ts">${time}</span>${tag}${escapeHtml(record.line)}</span>`;
}

function logVisible(record) {
  const instance = $("#log-instance").value;
  if (instance && String(record.instance) !== instance) return false;
  const filter = $("#log-filter").value.toLowerCase();
  return !filter || record.line.toLowerCase().includes(filter);
}

function appendLog(record) {
  state.logs.push(record);
  if (state.logs.length > 5000) state.logs.shift();
  if (!logVisible(record)) return;
  const view = $("#log-view");
  const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 40;
  view.insertAdjacentHTML("beforeend", logLine(record));
  if ($("#log-autoscroll").checked && atBottom) view.scrollTop = view.scrollHeight;
}

function renderLogs() {
  const view = $("#log-view");
  view.innerHTML = state.logs.filter(logVisible).map(logLine).join("");
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
      renderServers(data.servers || []);
      renderDownloads(data.downloads || []);
    } else if (type === "log") {
      appendLog(data);
    } else if (type === "hello") {
      state.logs = data.logs || [];
      renderLogs();
      renderServers(data.servers || []);
      (data.history || []).forEach((s) => {
        push("cpu", s.cpu.percent);
        push("mem", s.mem.percent);
        (s.gpus || []).forEach((g, i) => {
          if (g.busy !== null && g.busy !== undefined) push(`gpu${i}`, g.busy);
        });
        if (s.process_total && s.process_total.count) push("proc", s.process_total.cpu_percent);
      });
    }
  };
}

/* ----------------------------------------------------- huggingface hub */

function repoRow(repo) {
  return `<div class="repo" data-repo="${escapeHtml(repo.id)}">
      <div class="repo-head">
        <span class="repo-id">${escapeHtml(repo.id)}</span>
        <span class="repo-stats dim tiny">↓ ${repo.downloads.toLocaleString()} · ♥ ${repo.likes}</span>
      </div>
      <div class="repo-files" hidden></div>
    </div>`;
}

function fileRow(repo, file) {
  const disabled = file.is_projector ? "disabled" : "";
  const label = file.is_projector ? "projector" : "Download";
  return `<div class="file-row ${file.is_projector ? "projector" : ""}">
      <span class="fname">${escapeHtml(file.name)}</span>
      <span class="fsize">${bytes(file.size)}</span>
      <button class="btn small" data-repo="${escapeHtml(repo)}" data-path="${escapeHtml(file.path)}" ${disabled}>${label}</button>
    </div>`;
}

async function toggleRepo(card) {
  const box = card.querySelector(".repo-files");
  if (!box.hidden) { box.hidden = true; return; }
  box.hidden = false;
  if (box.dataset.loaded) return;
  box.innerHTML = '<p class="empty">loading files…</p>';
  try {
    const repo = card.dataset.repo;
    const data = await api(`/api/hub/files?repo=${encodeURIComponent(repo)}`);
    box.innerHTML = data.files.length
      ? data.files.map((f) => fileRow(repo, f)).join("")
      : '<p class="empty">no .gguf files in this repo</p>';
    box.dataset.loaded = "1";
    box.querySelectorAll("button[data-path]").forEach((button) => {
      button.onclick = async (event) => {
        event.stopPropagation();
        button.disabled = true;
        button.textContent = "starting…";
        try {
          await api("/api/hub/download", {
            method: "POST",
            body: JSON.stringify({ repo: button.dataset.repo, path: button.dataset.path }),
          });
          button.textContent = "queued";
        } catch (err) {
          button.textContent = "Download";
          button.disabled = false;
          alert(err.message);
        }
      };
    });
  } catch (err) {
    box.innerHTML = `<p class="empty">${escapeHtml(err.message)}</p>`;
  }
}

async function hubSearch(event) {
  event.preventDefault();
  const query = $("#hub-query").value.trim();
  if (!query) return;
  const results = $("#hub-results");
  results.innerHTML = '<p class="empty">searching…</p>';
  try {
    const data = await api(`/api/hub/search?q=${encodeURIComponent(query)}`);
    const rows = data.results;
    // a pasted "org/repo" is offered directly as well as searched for
    if (query.includes("/") && !rows.some((r) => r.id === query)) {
      rows.unshift({ id: query, downloads: 0, likes: 0 });
    }
    results.innerHTML = rows.length ? rows.map(repoRow).join("") : '<p class="empty">nothing found</p>';
    results.querySelectorAll(".repo").forEach((card) => {
      card.querySelector(".repo-head").onclick = () => toggleRepo(card);
    });
    if (rows.length === 1) toggleRepo(results.querySelector(".repo"));
  } catch (err) {
    results.innerHTML = `<p class="empty">${escapeHtml(err.message)}</p>`;
  }
}

function renderDownloads(downloads) {
  const host = $("#hub-downloads");
  if (!downloads.length) { host.innerHTML = ""; return; }

  // a finished download means a new file on disk — refresh the model list once
  const finished = downloads.filter((d) => d.status === "done").map((d) => d.id).join(",");
  if (state.finishedDownloads !== finished) {
    if (state.finishedDownloads !== undefined) loadModels();
    state.finishedDownloads = finished;
  }

  host.innerHTML = downloads.map((d) => {
    const speed = d.status === "downloading" ? ` · ${bytes(d.speed)}/s` : "";
    const eta = d.eta ? ` · ${duration(d.eta)} left` : "";
    const detail = d.status === "error" ? escapeHtml(d.error)
      : `${bytes(d.downloaded)} / ${bytes(d.total)}${speed}${eta}`;
    const action = ["queued", "downloading"].includes(d.status) ? "Cancel" : "Dismiss";
    return `<div class="dl" data-status="${d.status}">
        <div class="dl-head">
          <span class="dl-name">${escapeHtml(d.name)}</span>
          <span class="dim tiny">${escapeHtml(d.repo)}</span>
          <span class="dl-pct">${d.status === "done" ? "done" : d.percent + "%"}</span>
          <button class="btn small ghost" data-cancel="${d.id}">${action}</button>
        </div>
        <div class="bar"><div class="bar-fill" style="width:${d.percent}%"></div></div>
        <div class="dim tiny">${detail}</div>
      </div>`;
  }).join("");

  host.querySelectorAll("button[data-cancel]").forEach((button) => {
    button.onclick = () => api("/api/hub/cancel", {
      method: "POST",
      body: JSON.stringify({ id: +button.dataset.cancel }),
    }).catch((err) => alert(err.message));
  });
}

/* ------------------------------------------------------------------ chat */

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  const instance = $("#chat-instance").value;
  if (!text || !instance) return;
  input.value = "";

  const log = $("#chat-log");
  log.querySelector(".empty")?.remove();
  log.insertAdjacentHTML("beforeend", `<div class="msg user"></div>`);
  log.lastElementChild.textContent = text;
  log.insertAdjacentHTML("beforeend", `<div class="msg assistant"></div>`);
  const bubble = log.lastElementChild;
  log.scrollTop = log.scrollHeight;

  const body = { instance: +instance, messages: [{ role: "user", content: text }], max_tokens: 512 };
  const modelSelect = $("#chat-model");
  if (!modelSelect.hidden && modelSelect.value) body.model = modelSelect.value;

  const started = performance.now();
  let chunks = 0;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
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
          if (delta) { bubble.textContent += delta; chunks++; log.scrollTop = log.scrollHeight; }
        } catch { /* partial frame */ }
      }
    }
    const seconds = (performance.now() - started) / 1000;
    $("#chat-stats").textContent = `${chunks} chunks in ${seconds.toFixed(1)}s`;
  } catch (err) {
    bubble.classList.add("err");
    bubble.textContent = `error: ${err.message}`;
  }
}

/* ------------------------------------------------------------------ init */

async function init() {
  state.info = await api("/api/info");
  const system = state.info.system;
  $("#hostname").textContent = system.hostname;
  $("#cpu-model").textContent =
    `${system.cpu_model} · ${system.cpu_count} threads · ${bytes(system.mem_total)}`;
  setParams(state.info.config.defaults);
  if (!$("#p-models-dir").value) $("#p-models-dir").value = state.info.download_dir;

  $("#d-system").innerHTML = Object.entries(system).map(([k, v]) => {
    const text = k === "gpus"
      ? (v.length ? v.map((g) => `${g.vendor} ${g.name} (${g.driver})`).join(", ") : "none")
      : (k === "mem_total" ? bytes(v) : v);
    return `<span class="k">${k}</span><span class="v">${escapeHtml(text)}</span>`;
  }).join("");
  await loadExternal();

  renderAutostart((await api("/api/autostart")).entries);

  const downloads = await api("/api/hub/downloads");
  $("#hub-dest").textContent = `downloads are saved to ${downloads.dest}`;
  renderDownloads(downloads.downloads);

  await loadModels();
  const last = state.info.config.last_model;
  if (last && state.models.some((m) => m.path === last)) selectModel(last);
  setMode("single");
  connect();
}

$("#btn-start").onclick = async () => {
  const button = $("#btn-start");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    await api("/api/server/start", {
      method: "POST",
      body: JSON.stringify({
        model_path: state.mode === "router" ? "" : state.selected,
        params: getParams(),
      }),
    });
    document.querySelector('.tab[data-tab="logs"]').click();
  } catch (err) {
    alert(`Failed to start: ${err.message}`);
  } finally {
    button.textContent = label;
    updateStartButton();
  }
};

$("#btn-stop-all").onclick = async () => {
  for (const server of state.servers.filter((s) => s.pid)) {
    await api("/api/server/stop", { method: "POST", body: JSON.stringify({ id: server.id }) })
      .catch((err) => alert(err.message));
  }
};

$("#btn-save-autostart").onclick = async () => {
  const button = $("#btn-save-autostart");
  const running = state.servers.filter((s) => s.pid).length;
  if (!running && !confirm("No servers are running — save an empty startup set?")) return;
  button.disabled = true;
  try {
    const data = await api("/api/autostart", { method: "POST", body: "{}" });
    renderAutostart(data.entries);
    button.textContent = "Saved ✓";
    setTimeout(() => { button.textContent = "Save as startup"; }, 2000);
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
  }
};

$("#btn-clear-autostart").onclick = async () => {
  const data = await api("/api/autostart", {
    method: "POST", body: JSON.stringify({ entries: [] }),
  }).catch((err) => { alert(err.message); return null; });
  if (data) renderAutostart(data.entries);
};

$("#btn-rescan").onclick = loadModels;
$("#btn-toggle-dirs").onclick = () => {
  $("#model-dirs").classList.toggle("hidden");
  $("#btn-toggle-dirs").classList.toggle("on", !$("#model-dirs").classList.contains("hidden"));
};
$("#btn-toggle-filter").onclick = () => {
  const input = $("#model-filter");
  const showing = input.classList.toggle("hidden");
  $("#btn-toggle-filter").classList.toggle("on", !showing);
  if (showing) { input.value = ""; renderModels(); } else { input.focus(); }
};
$("#model-filter").oninput = renderModels;
$("#log-filter").oninput = renderLogs;
$("#log-instance").onchange = renderLogs;
$("#chat-instance").onchange = updateChatModels;
$("#btn-clear-log").onclick = () => { state.logs = []; renderLogs(); };
$("#chat-form").onsubmit = sendChat;
$("#hub-form").onsubmit = hubSearch;
$("#chat-input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); }
};
$$(".mode").forEach((button) => { button.onclick = () => setMode(button.dataset.mode); });
$$(".params input, .params select").forEach((el) => {
  el.addEventListener("input", () => { updatePreview(); updateStartButton(); });
});
$("#btn-manage-refresh").onclick = () => { loadModels(); loadExternal(); };
$$(".tab").forEach((tab) => {
  tab.onclick = () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab.dataset.tab}`));
    // process list is cheap and goes stale quickly — refresh when it is opened
    if (tab.dataset.tab === "manage") loadExternal();
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

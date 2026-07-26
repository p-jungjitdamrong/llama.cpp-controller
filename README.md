# llama-controller

A small web control panel that wraps `llama-server` from llama.cpp: pick a GGUF,
start it with the flags you want, run several models at once, download new ones
from Hugging Face, and watch CPU / RAM / GPU and the server logs live — without
touching the terminal.

Runs on the machine that hosts llama.cpp. Four dependencies, no build step, no
database — the dashboard is plain HTML/CSS/JS served by the same process.

```
                          ┌─ llama-server :8090  (one model)
browser ──HTTP/WS──> llamactl ─ llama-server :8091  (another model)
                     :8080    └─ llama-server :8100 --models-dir  (router)
                        │
                        └── /proc + /sys/class/drm (CPU, RAM, GPU)
```

## What it does

- **Model list** — scans configured directories for `.gguf` and reads each file's
  GGUF header for architecture, quantisation, training context and layer count.
  Vocab-only test files are filtered out; multi-part shards show once.
- **Run several models at once** — each model gets its own llama-server process
  on its own port. Ask for a port that is taken and the next free one is used,
  so starting a second model is one click. Every server is listed with its
  state, endpoint URL, CPU, memory and tokens/second, and can be stopped or
  restarted on its own.
- **Router mode** — the alternative: one llama-server started with
  `--models-dir` hosts a whole directory (plus anything in the Hugging Face
  cache) on a single port and loads models on demand, up to `--models-max` at a
  time. The dashboard lists what the router knows about, shows which models are
  resident, and can load or unload them explicitly. The launch flags are written
  to an INI preset the router applies to every child it spawns — without one
  those children start with plain defaults and run on the CPU.
- **Manage tab** — every discovered model in one table with quantisation,
  architecture, size, training context, whether it is running or in the startup
  set, and how much disk is left. Models can be started or deleted from here;
  deletion refuses anything outside the configured directories or currently
  being served. llama-server processes started outside the controller are listed
  with a Kill button (a router's own children are excluded — unload the model
  instead).
- **Launch flags** (`--host`, `--port`, `-ngl`, `-c`, `-t`, `-np`, `-b`, `-fa`,
  `--mlock`, `--no-mmap`, `--jinja`, plus free-form extra args) are editable in
  the UI and saved per model, so the next start of that model reuses them. The
  bind address defaults to `0.0.0.0` so the model API is reachable from other
  machines, and each server shows the exact URL to point a client at.
- **Startup set** — *Save as startup* records exactly which servers are running
  right now, and the controller brings them back in the same order the next time
  it starts, waiting for each to load before starting the next. Combined with the
  systemd unit that means the box comes back from a reboot serving models with
  nobody logged in.
- **Download from Hugging Face** — search the Hub (or paste an `org/repo`), see
  every GGUF file with its size, and download with a live progress bar. Partial
  transfers resume with a Range request; finished downloads appear in the model
  list automatically. No token and no `huggingface_hub` dependency.
- **Monitoring at 1 Hz** — total and per-core CPU, memory and swap, every
  detected GPU (busy percentage, memory, temperature, power, clock), and the
  llama-server process's own CPU, RSS and thread count.
- **Live log** — every server's stdout/stderr streamed over one websocket, tagged
  with the model it came from, filterable by server and by text. Prompt and
  generation tokens/second are parsed out of the timing lines.
- **Test chat** — a minimal streaming chat box that proxies to a chosen server's
  OpenAI-compatible endpoint (and, for a router, a chosen model) to confirm it
  actually answers.
- **Leaves other servers alone** — llama-server processes it did not start are
  listed read-only under *Details* and are never signalled. The one exception is
  a llama-server left on its own port by a previous run of the controller (same
  binary, same port): that one is cleaned up before the next start, so a crashed
  controller cannot lock you out of the port.

## Hardware support

| | How it is read | What you get |
| --- | --- | --- |
| AMD (`amdgpu`) | `/sys/class/drm/cardN/device` | busy %, VRAM, GTT, temp, clock |
| NVIDIA | `nvidia-smi` if on PATH | busy %, VRAM, temp, power, clock |
| Intel (`i915`/`xe`) | same sysfs tree | memory and clock; busy % is not exposed |
| No GPU / unknown | — | card shows "no supported GPU detected" |

Several GPUs, and several vendors at once, each get their own card. CPU and
memory come from `/proc` on Linux; on other platforms the controller falls back
to `psutil` when it is installed (`pip install psutil`) and otherwise reports
what it can.

## Install

On the llama.cpp host:

```bash
python3 -m venv ~/llama_controller_venv
~/llama_controller_venv/bin/pip install -r requirements.txt
```

On Debian/Ubuntu, `python3 -m venv` may first need `sudo apt install python3-venv`
(or the versioned package, e.g. `python3.14-venv`).

## Run

As a service, which is what you want on a box that should come back after a
reboot:

```bash
sudo ./scripts/install-service.sh
```

It fills the unit template in `systemd/` with the current user, paths and venv,
installs it to `/etc/systemd/system/llama-controller.service`, then enables and
starts it. The service runs unprivileged as the user who owns the checkout —
sudo is only needed to write the unit file. `PORT=`, `HOST=`, `LLAMA_PORT=` and
`PYTHON=` override the defaults. After that it is ordinary systemd:

```bash
sudo systemctl restart llama-controller
systemctl status llama-controller
journalctl -u llama-controller -f
```

Every llama-server it starts lives in the service's cgroup, so `systemctl stop`
takes the models down with it and leaves nothing orphaned.

Without systemd, or while developing:

```bash
./scripts/ctl.sh start        # background, logs to controller.log
./scripts/ctl.sh logs 50
./scripts/ctl.sh stop
```

Or in the foreground:

```bash
~/llama_controller_venv/bin/python -m llamactl --port 8080 --llama-port 8090
```

Then open `http://<host>:8080`.

## Configuration

`config.json` is written next to the package on first run and updated whenever
you start a model. Edit it directly or use `POST /api/config`.

| Key | Meaning |
| --- | --- |
| `llama_server_bin` | path to the `llama-server` binary |
| `model_dirs` | directories scanned for `.gguf` (3 levels deep, hidden dirs skipped) |
| `controller_host` / `controller_port` | where the dashboard binds |
| `defaults` | launch flags for a model with no saved preset, including `host` and `port` for llama-server |
| `presets` | per-model launch flags, saved automatically on start |
| `autostart` | servers to bring up on startup, in order — written by *Save as startup* |

The same defaults can be set on the command line: `--host` / `--port` for the
dashboard, `--llama-host` / `--llama-port` for the model server it spawns.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/info` | host details, config, foreign llama-server processes |
| GET | `/api/models` | discovered GGUF files with metadata, disk usage, and what was skipped |
| POST | `/api/models/delete` | `{path, confirm: true}` — only inside the model dirs, never a model in use |
| GET | `/api/processes` | llama-server processes, ours and foreign |
| POST | `/api/processes/kill` | `{pid}` — foreign llama-server processes only |
| GET | `/api/status` | every instance: state, argv, endpoint, throughput |
| POST | `/api/server/start` | `{model_path, params}`; `params.mode: "router"` starts a router |
| POST | `/api/server/stop` · `restart` · `remove` | `{id}` |
| POST | `/api/server/clear-port` | kill a llama-server orphaned by an earlier controller run |
| POST | `/api/router/load` · `unload` · `refresh` | `{id, model}` — drive a router instance |
| GET · POST | `/api/autostart` | read the startup set; POST `{}` saves what is running, `{entries: []}` clears |
| GET | `/api/hub/search?q=` · `/api/hub/files?repo=` | browse the Hugging Face Hub |
| POST | `/api/hub/download` · `/api/hub/cancel` | `{repo, path}` / `{id}` |
| GET | `/api/hub/downloads` | progress of every transfer |
| GET | `/api/logs?limit=&instance=` | recent log lines, optionally for one server |
| GET | `/api/metrics/history` | recent metric samples |
| WS | `/ws` | 1 Hz metrics, server states and downloads; log lines as they arrive |
| POST | `/api/chat` | streaming passthrough; `{instance, model?, messages}` |
| ANY | `/proxy/{id}/{path}` | passthrough to one server (`/slots`, `/metrics`, …) |

Interactive docs at `/api/docs`.

## Notes and limits

- **Nothing stops you running out of memory.** Each extra model is a real
  process holding real weights; the controller shows free memory but will not
  refuse a start. Router mode is the gentler option — it caps how many models
  are resident with `--models-max` and loads the rest on demand.
- **Two ways to serve several models, pick per situation.** Separate instances
  give each model its own flags, port and log, and they stay warm. A router
  gives one endpoint and one port for many models, at the cost of a load pause
  when a request hits a model that is not resident.
- **On an AMD APU**, weights may land in GTT (shared system RAM) rather than the
  small VRAM carve-out, depending on the backend. The card shows whichever of
  the two is actually carrying the model and labels it.
- **No authentication.** It can start processes and both it and llama-server
  bind `0.0.0.0` by default; keep them on a trusted network, or set the bind
  host to `127.0.0.1` and reach the dashboard over an SSH tunnel:
  `ssh -L 8080:127.0.0.1:8080 user@host`.

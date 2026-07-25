# llama-controller

A small web control panel that wraps `llama-server` from llama.cpp: pick a GGUF,
start it with the flags you want, watch CPU / RAM / GPU and the server log live,
and switch models without touching the terminal.

Runs on the machine that hosts llama.cpp. Four dependencies, no build step, no
database — the dashboard is plain HTML/CSS/JS served by the same process.

```
browser ──HTTP/WebSocket──> llamactl (FastAPI, :8080) ──spawns──> llama-server (:8090)
                                   │
                                   └── /proc + /sys/class/drm (CPU, RAM, GPU)
```

## What it does

- **Model list** — scans configured directories for `.gguf` and reads each file's
  GGUF header for architecture, quantisation, training context and layer count.
  Vocab-only test files are filtered out; multi-part shards show once.
- **Start / stop / restart** — one model at a time. Launch flags (`--host`,
  `--port`, `-ngl`, `-c`, `-t`, `-np`, `-b`, `-fa`, `--mlock`, `--no-mmap`,
  `--jinja`, plus free-form extra args) are editable in the UI and saved per
  model, so the next start of that model reuses them. The bind address defaults
  to `0.0.0.0` so the model API is reachable from other machines, and the header
  shows the exact URL to point a client at.
- **Monitoring at 1 Hz** — total and per-core CPU, memory and swap, every
  detected GPU (busy percentage, memory, temperature, power, clock), and the
  llama-server process's own CPU, RSS and thread count.
- **Live log** — llama-server's stdout/stderr streamed over a websocket, with a
  filter box and level colouring. Prompt and generation tokens/second are parsed
  out of the timing lines and shown under *Details*.
- **Test chat** — a minimal streaming chat box that proxies to the running
  server's OpenAI-compatible endpoint, to confirm a model actually answers.
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

To start it at boot, install the user service:

```bash
mkdir -p ~/.config/systemd/user && cp systemd/llama-controller.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now llama-controller
sudo loginctl enable-linger "$USER"   # so it survives logout
```

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

The same defaults can be set on the command line: `--host` / `--port` for the
dashboard, `--llama-host` / `--llama-port` for the model server it spawns.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/info` | host details, config, foreign llama-server processes |
| GET | `/api/models` | discovered GGUF files with metadata |
| GET | `/api/status` | supervisor state, argv, throughput stats |
| POST | `/api/server/start` | `{model_path, params}` |
| POST | `/api/server/stop` · `/api/server/restart` | lifecycle |
| POST | `/api/server/clear-port` | kill a llama-server orphaned by an earlier controller run |
| GET | `/api/logs?limit=` | recent log lines |
| GET | `/api/metrics/history` | recent metric samples |
| WS | `/ws` | 1 Hz metrics + status, and log lines as they arrive |
| POST | `/api/chat` | streaming passthrough to `/v1/chat/completions` |
| ANY | `/proxy/{path}` | passthrough to llama-server (`/slots`, `/metrics`, …) |

Interactive docs at `/api/docs`.

## Notes and limits

- **One model at a time.** Switching stops the current server and starts a new
  one. On a 13 GB laptop that is the honest constraint; recent llama.cpp also has
  a built-in router mode (`--models-dir`, `--models-max`) if you have the RAM to
  keep several loaded — this controller does not use it.
- **On an AMD APU**, weights may land in GTT (shared system RAM) rather than the
  small VRAM carve-out, depending on the backend. The card shows whichever of
  the two is actually carrying the model and labels it.
- **No authentication.** It can start processes and both it and llama-server
  bind `0.0.0.0` by default; keep them on a trusted network, or set the bind
  host to `127.0.0.1` and reach the dashboard over an SSH tunnel:
  `ssh -L 8080:127.0.0.1:8080 user@host`.

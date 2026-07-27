# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0.0 the API may change between minor versions; anything that would break
an existing `config.json` is called out under **Changed**.

## [Unreleased]

### Added
- **Tune** in the Manage tab: each candidate configuration is started for real,
  asked the same question, and timed, with prompt processing and generation
  reported separately. Any row can be saved as that model's settings, which then
  apply when it is started on its own and when a router loads it. Sweeps GPU
  offload and thread count by default, because those are the two that moved the
  numbers on modest hardware.
- Each model's saved settings are shown in the Manage tab, marked when they are
  the model's own rather than the controller's defaults. Until now the only way
  to see them was to select the model and read the Launch panel.
- Optional access control: a token checked on the API, the websocket and the
  proxy routes, usable as `Authorization: Bearer …` by any OpenAI-compatible
  client. Requests from the machine itself are exempt unless you say otherwise,
  the token is readable in `config.json`, and `--no-auth` starts without it —
  three ways back in if a token is lost.
- A Settings tab for the binary path, model directories, defaults for new models
  and retention, with the paths validated before they are saved.
- Router presets now carry a section per model, built from the settings each
  model was last started with, so a model tuned in "One model" mode keeps that
  tuning when a router loads it. Only the values that differ from the shared
  `[*]` section are written.

### Fixed
- Benchmarking an embedding model failed on every configuration: those servers
  have no chat endpoint. They are now detected from the GGUF architecture or the
  `--embedding` flag and probed with `/v1/embeddings`, ranked by latency instead
  of tokens per second. On this hardware the GPU turns out to be twelve times
  faster at it than the CPU.
- Benchmarking a model that was already serving failed once per configuration
  with "already running". The clash is now reported once, before the sweep, with
  the option to stop the running copy and start it again afterwards.

## [0.3.0] — 2026-07-27

### Added
- Plain-language help under every launch flag, with a `?` toggle in the Launch
  panel that remembers your choice. Written around what surprises people: that
  `-np` divides the context rather than multiplying it, and that GPU offload is
  not automatically faster on an integrated GPU.
- `scripts/smoke-test.py`, which walks the app's own route table and calls every
  endpoint with a harmless body, so a helper deleted by a careless edit cannot
  hide behind Python's late name binding.
- Screenshots in the README, and an architecture diagram.

### Fixed
- Every endpoint that resolves a server by id — stop, restart, remove, chat,
  the router controls — raised `NameError` after a helper was removed with the
  agent code.
- The test chat looked frozen with a reasoning model: it streams thinking in
  `reasoning_content` and leaves `content` empty until it is done. Thinking now
  streams into a collapsible block that folds away when the answer starts.
- Metric cards stayed empty for a second after every page load, ignoring the
  five minutes of history the websocket handshake already carries.

### Changed
- Licence is now Apache-2.0 rather than MIT, for the explicit patent grant.
  Source files carry an SPDX identifier; `NOTICE` records that llama.cpp remains
  MIT and is never redistributed here.
- The agent prototype was removed; it lives on the `agent-phase1` branch and
  belongs in a project of its own.

## [0.2.0] — 2026-07-26

### Added
- **Several models at once.** One `llama-server` process per model, ports
  allocated automatically, each with its own state, endpoint, logs and controls.
- **Router mode** driving llama.cpp's own `--models-dir` server, including the
  model list, resident state and explicit load/unload.
- **Per-model GPU usage** from DRM fdinfo (AMD/Intel) or `nvidia-smi`: VRAM, GTT
  and the share of the GPU each model is using, not just the card total.
- **Hugging Face downloads** — search, browse quantisations by size, resumable
  transfers with live progress.
- **Manage tab** listing every discovered GGUF with quantisation, architecture,
  size, context, running state and disk headroom, plus delete.
- **Startup set** — save what is running and have it come back on boot, with a
  systemd unit installed by `scripts/install-service.sh`.
- **Memory guard** measuring weights plus KV cache against what is already
  promised to running instances, with `MemoryMax` in the unit as a backstop.
- `scripts/bench-models.py`, which ranks models by what an agent loop depends on:
  tool-call support, correctness, speed and prompt-cache reuse.

### Fixed
- Router children ran on the CPU. llama.cpp spawns a child per loaded model and,
  without a preset, starts it with plain defaults and no `-ngl`. The launch flags
  are now written to an INI preset passed with `--models-preset`.
- A llama-server orphaned by a crashed controller kept its port; the next start
  now clears it, matched strictly on binary and port.
- `pkill -f` over SSH matched the remote shell's own command line and killed the
  session; `scripts/ctl.sh` uses a pid file.

## [0.1.0] — 2026-07-25

### Added
- First working panel: start and stop one `llama-server`, live CPU, memory
  and GPU metrics from `/proc` and `/sys`, streamed logs, GGUF discovery with
  header parsing, and a streaming test chat.

[Unreleased]: https://github.com/p-jungjitdamrong/llama.cpp-controller/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/p-jungjitdamrong/llama.cpp-controller/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/p-jungjitdamrong/llama.cpp-controller/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/p-jungjitdamrong/llama.cpp-controller/releases/tag/v0.1.0

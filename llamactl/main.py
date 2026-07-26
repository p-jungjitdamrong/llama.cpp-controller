"""Entry point: python -m llamactl [--host …] [--port …]"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import argparse
import sys

import uvicorn

from .api import create_app
from .config import CONFIG_PATH, Config


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    parser = argparse.ArgumentParser(prog="llamactl", description="llama.cpp control panel")
    parser.add_argument("--host", default=cfg.controller_host)
    parser.add_argument("--port", type=int, default=cfg.controller_port)
    parser.add_argument("--llama-port", type=int, default=cfg.defaults.port,
                        help="default port for the supervised llama-server")
    parser.add_argument("--llama-host", default=cfg.defaults.host,
                        help="default bind address for llama-server (0.0.0.0 = reachable "
                             "from other machines)")
    parser.add_argument("--bin", default=cfg.llama_server_bin, help="path to llama-server")
    parser.add_argument("--models-dir", action="append", default=None,
                        help="directory to scan for .gguf (repeatable)")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--no-auth", action="store_true",
                        help="ignore the saved token for this run — the way back "
                             "in if you lock yourself out")
    args = parser.parse_args(argv)

    cfg.controller_host, cfg.controller_port = args.host, args.port
    cfg.llama_server_bin = args.bin
    cfg.defaults.port, cfg.defaults.host = args.llama_port, args.llama_host
    if args.models_dir:
        cfg.model_dirs = args.models_dir
    cfg.save()
    if args.no_auth:
        cfg.auth_enabled = False   # in memory only; the saved token is untouched
        print("auth disabled for this run (--no-auth)")

    if not cfg.server_bin.is_file():
        print(f"warning: llama-server not found at {cfg.server_bin}", file=sys.stderr)

    print(f"config       {CONFIG_PATH}")
    print(f"llama-server {cfg.server_bin}  ->  {cfg.defaults.host}:{cfg.defaults.port}")
    print(f"dashboard    http://{args.host}:{args.port}")
    # without a graceful timeout, an open dashboard websocket keeps uvicorn from
    # shutting down, the supervisor never gets to stop llama-server, and the
    # child is orphaned holding its port
    uvicorn.run(create_app(cfg), host=args.host, port=args.port,
                log_level=args.log_level, timeout_graceful_shutdown=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

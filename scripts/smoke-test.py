#!/usr/bin/env python3
"""Hit every route once against a running controller.

Python only binds names at call time, so a helper deleted by a careless edit
looks fine until someone clicks the button that needs it. This walks the app's
own route table and calls each one, and treats any 500 as a failure — 404 and
409 are fine, they mean the handler ran and said no.

Usage: python3 scripts/smoke-test.py [http://127.0.0.1:8080]
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

# bodies that reach the handler without changing anything: a nonexistent id
SAFE_BODIES = {
    "/api/server/stop": {"id": 999999},
    "/api/server/restart": {"id": 999999},
    "/api/server/remove": {"id": 999999},
    "/api/router/load": {"id": 999999, "model": "nope"},
    "/api/router/unload": {"id": 999999, "model": "nope"},
    "/api/router/refresh": {"id": 999999},
    "/api/chat": {"instance": 999999, "messages": [{"role": "user", "content": "hi"}]},
    "/api/models/delete": {"path": "/nonexistent/model.gguf", "confirm": True},
    "/api/processes/kill": {"pid": 999999},
    "/api/hub/download": {"repo": "", "path": ""},
    "/api/hub/cancel": {"id": 999999},
    "/api/server/clear-port": {"port": 65534},
}
SKIP = {"/api/autostart", "/api/config"}  # POST here would rewrite saved settings
QUERY = {"/api/hub/search": "?q=test", "/api/hub/files": "?repo=ggml-org/models"}


def call(method: str, url: str, body: dict | None) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")[:120]
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
    try:
        spec = json.loads(urllib.request.urlopen(f"{base}/openapi.json", timeout=15).read())
    except Exception as exc:
        print(f"cannot reach {base}: {exc}")
        return 2

    failures = []
    for path, methods in sorted(spec["paths"].items()):
        if "{" in path or path in SKIP:
            continue
        for method in methods:
            if method.upper() not in ("GET", "POST"):
                continue
            url = base + path + (QUERY.get(path, "") if method == "get" else "")
            body = SAFE_BODIES.get(path) if method == "post" else None
            if method == "post" and body is None:
                continue  # no safe body known — skip rather than guess
            status, detail = call(method.upper(), url, body)
            ok = 200 <= status < 500 and status != 0
            print(f"  {'ok ' if ok else 'FAIL'} {method.upper():4} {path:34} {status} {detail[:70]}")
            if not ok:
                failures.append((method.upper(), path, status, detail))

    print(f"\n{len(failures)} failing route(s)")
    for method, path, status, detail in failures:
        print(f"  {method} {path} -> {status} {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

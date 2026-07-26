#!/usr/bin/env python3
"""Rank local GGUF models for agent work.

For each model this starts an instance through the controller, then measures the
things an agent loop actually depends on:

  * does the chat template support tools at all
  * does the model emit a correct tool call for an unambiguous question
  * does it produce a sensible answer once the tool result comes back
  * how fast it reads the prompt and writes tokens
  * how much a repeated prefix costs — the loop re-sends the whole history every
    step, so prompt-cache reuse matters more than raw generation speed

Usage: python3 scripts/bench-agent-models.py [--controller URL] [--ctx 4096]
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Look up the current share price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "e.g. AAPL"},
                    "currency": {"type": "string", "enum": ["USD", "THB"]},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Full-text search over internal documents",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

QUESTION = "What is Apple's share price right now, in Thai baht?"
EXPECTED_TOOL = "get_stock_price"
TOOL_RESULT = json.dumps({"ticker": "AAPL", "price": 8421.50, "currency": "THB"})


def post(url: str, payload: dict, timeout: float = 600) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def start(controller: str, path: str, ctx: int, port: int) -> dict:
    return post(f"{controller}/api/server/start", {
        "model_path": path,
        "params": {"host": "127.0.0.1", "port": port, "ctx_size": ctx,
                   "n_gpu_layers": 99, "parallel": 1},
    })


def wait_ready(controller: str, instance_id: int, timeout: float = 420) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for instance in get(f"{controller}/api/status")["instances"]:
            if instance["id"] != instance_id:
                continue
            if instance["state"] == "ready":
                return instance
            if instance["state"] in ("error", "stopped"):
                raise RuntimeError(instance["last_error"] or "server exited")
        time.sleep(2)
    raise TimeoutError("model did not become ready")


def chat(url: str, messages: list, tools: list | None = None,
         max_tokens: int = 200) -> tuple[dict, dict]:
    payload: dict = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if tools:
        payload["tools"] = tools
    started = time.monotonic()
    body = post(f"{url}/v1/chat/completions", payload)
    elapsed = time.monotonic() - started
    choice = body["choices"][0]
    timings = body.get("timings") or {}
    timings["wall"] = elapsed
    return choice, timings


def bench_model(controller: str, model: dict, ctx: int, port: int) -> dict:
    row: dict = {"name": model["name"], "size": model["size"], "arch": model.get("architecture", "")}
    started = time.monotonic()
    instance = start(controller, model["path"], ctx, port)
    instance_id = instance["id"]
    try:
        wait_ready(controller, instance_id)
        row["load_s"] = round(time.monotonic() - started, 1)
        url = f"http://127.0.0.1:{port}"

        caps = get(f"{url}/props").get("chat_template_caps", {})
        row["tools_supported"] = bool(caps.get("supports_tool_calls"))
        row["parallel_tools"] = bool(caps.get("supports_parallel_tool_calls"))

        messages = [
            {"role": "system", "content": "You are a precise assistant. Use the tools when they apply."},
            {"role": "user", "content": QUESTION},
        ]
        choice, timings = chat(url, messages, TOOLS)
        calls = choice["message"].get("tool_calls") or []
        row["called_tool"] = calls[0]["function"]["name"] if calls else None
        row["tool_ok"] = bool(calls) and calls[0]["function"]["name"] == EXPECTED_TOOL
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"])
                row["args_ok"] = args.get("ticker", "").upper() == "AAPL"
            except (json.JSONDecodeError, AttributeError):
                row["args_ok"] = False
        row["step1_s"] = round(timings["wall"], 1)
        row["prefill_tps"] = round(timings.get("prompt_per_second") or 0, 1)
        row["gen_tps"] = round(timings.get("predicted_per_second") or 0, 1)

        # second leg of the loop: hand the tool result back and ask for the answer
        if calls:
            messages.append({"role": "assistant", "tool_calls": calls})
            messages.append({"role": "tool", "tool_call_id": calls[0].get("id", "0"),
                             "name": EXPECTED_TOOL, "content": TOOL_RESULT})
            choice2, timings2 = chat(url, messages, TOOLS)
            answer = (choice2["message"].get("content") or "").strip()
            row["answered"] = "8421" in answer.replace(",", "") or "8,421" in answer
            row["answer"] = answer[:80].replace("\n", " ")
            row["step2_s"] = round(timings2["wall"], 1)
            # third call over the same prefix — shows what the KV cache saves
            messages.append({"role": "user", "content": "And in US dollars?"})
            _, timings3 = chat(url, messages, TOOLS, max_tokens=60)
            row["cached_prefill_tps"] = round(timings3.get("prompt_per_second") or 0, 1)
            row["cache_n"] = timings3.get("cache_n", 0)
        row["loop_s"] = round(row.get("step1_s", 0) + row.get("step2_s", 0), 1)
    except Exception as exc:  # a model that cannot run is a result too
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            post(f"{controller}/api/server/stop", {"id": instance_id}, timeout=120)
            post(f"{controller}/api/server/remove", {"id": instance_id}, timeout=30)
        except Exception:
            pass
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="http://127.0.0.1:8080")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--max-size-gb", type=float, default=6.0,
                        help="skip models larger than this")
    parser.add_argument("--only", default="", help="substring filter on the file name")
    parser.add_argument("--out", default="/tmp/agent-bench.json")
    args = parser.parse_args()

    models = get(f"{args.controller}/api/models")["models"]
    models = [m for m in models
              if m["size"] <= args.max_size_gb * 2**30
              and args.only.lower() in m["name"].lower()
              and "embed" not in m["name"].lower()]
    print(f"benchmarking {len(models)} models at ctx={args.ctx}\n")

    rows = []
    for model in sorted(models, key=lambda m: m["size"]):
        print(f"→ {model['name']} ({model['size'] / 2**30:.2f} GB)", flush=True)
        row = bench_model(args.controller, model, args.ctx, args.port)
        rows.append(row)
        if "error" in row:
            print(f"   failed: {row['error']}\n", flush=True)
        else:
            print(f"   load {row['load_s']}s · tool={row['called_tool']} "
                  f"ok={row.get('tool_ok')} · gen {row['gen_tps']} tok/s · "
                  f"loop {row.get('loop_s')}s\n", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    header = f"{'model':44} {'GB':>5} {'load':>5} {'tool':>5} {'args':>5} {'ans':>4} {'gen t/s':>8} {'loop s':>7} {'cached pp':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -(r.get("gen_tps") or 0)):
        if "error" in row:
            print(f"{row['name'][:44]:44} {row['size'] / 2**30:5.2f}  failed: {row['error'][:50]}")
            continue
        print(f"{row['name'][:44]:44} {row['size'] / 2**30:5.2f} {row['load_s']:5.1f} "
              f"{str(row.get('tool_ok')):>5} {str(row.get('args_ok')):>5} "
              f"{str(row.get('answered')):>4} {row['gen_tps']:8.1f} "
              f"{row.get('loop_s', 0):7.1f} {row.get('cached_prefill_tps', 0):10.1f}")
    print(f"\nfull results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

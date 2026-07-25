#!/usr/bin/env bash
# Start/stop/restart the controller on the machine it runs on.
#   ./scripts/ctl.sh start|stop|restart|status|logs
set -uo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/llama_controller_venv}"
PYTHON="$VENV/bin/python"
PID_FILE="$APP_DIR/controller.pid"
LOG_FILE="$APP_DIR/controller.log"
PORT="${PORT:-8080}"
LLAMA_PORT="${LLAMA_PORT:-8090}"

running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-status}" in
  start)
    if running; then echo "already running (pid $(cat "$PID_FILE"))"; exit 0; fi
    # installed as a service? then that owns the port — don't start a second one
    if command -v systemctl >/dev/null && systemctl is-active --quiet llama-controller 2>/dev/null; then
      echo "llama-controller.service is already running — use 'sudo systemctl restart llama-controller'"
      exit 1
    fi
    cd "$APP_DIR"
    setsid "$PYTHON" -m llamactl --port "$PORT" --llama-port "$LLAMA_PORT" \
      </dev/null >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    running && echo "started (pid $(cat "$PID_FILE")) on port $PORT" || { echo "failed:"; tail -20 "$LOG_FILE"; exit 1; }
    ;;
  stop)
    if running; then
      pid=$(cat "$PID_FILE")
      kill "$pid" 2>/dev/null
      # give it time to shut its llama-server child down cleanly before SIGKILL,
      # otherwise the child is orphaned and keeps holding the llama port
      for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
      kill -9 "$pid" 2>/dev/null
      echo "stopped"
    else
      echo "not running"
    fi
    rm -f "$PID_FILE"
    ;;
  restart) "$0" stop; "$0" start ;;
  status)
    running && echo "running (pid $(cat "$PID_FILE"))" || echo "not running"
    ;;
  logs) tail -n "${2:-50}" "$LOG_FILE" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}"; exit 1 ;;
esac

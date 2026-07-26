#!/usr/bin/env bash
# Install the controller as a systemd system service running as the current user.
# Run it on the machine that hosts llama.cpp:
#
#   sudo ./scripts/install-service.sh            # install, enable, start
#   PORT=9000 sudo ./scripts/install-service.sh  # different dashboard port
#
# sudo is needed only to write the unit file; the service itself runs unprivileged.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_NAME="llama-controller.service"
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"

# who owns the checkout — that is who the service should run as, even under sudo
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "$RUN_USER")"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
PYTHON="${PYTHON:-$RUN_HOME/llama_controller_venv/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
LLAMA_PORT="${LLAMA_PORT:-8090}"

if [[ $EUID -ne 0 ]]; then
  echo "needs root to write $UNIT_PATH — re-run with sudo" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "python not found at $PYTHON — create the venv first, or set PYTHON=" >&2
  exit 1
fi

sed -e "s|__USER__|$RUN_USER|" \
    -e "s|__GROUP__|$RUN_GROUP|" \
    -e "s|__HOME__|$RUN_HOME|" \
    -e "s|__APP_DIR__|$APP_DIR|" \
    -e "s|__PYTHON__|$PYTHON|" \
    -e "s|__HOST__|$HOST|" \
    -e "s|__PORT__|$PORT|" \
    -e "s|__LLAMA_PORT__|$LLAMA_PORT|" \
    "$APP_DIR/systemd/$UNIT_NAME" > "$UNIT_PATH"

systemctl daemon-reload
systemctl enable "$UNIT_NAME"
# restart, not just start — reinstalling over a running service must pick up both
# the new unit file and the new code
systemctl restart "$UNIT_NAME"
sleep 2
systemctl --no-pager --lines=0 status "$UNIT_NAME" || true

echo
echo "installed  $UNIT_PATH"
echo "dashboard  http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
echo "logs       journalctl -u $UNIT_NAME -f"

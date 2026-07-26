#!/usr/bin/env bash
# Push the controller to the llama.cpp host and (re)start it under systemd --user.
# Usage: ./deploy.sh [user@host] [remote-dir]
set -euo pipefail

HOST="${1:-${LLAMACTL_HOST:?pass user@host, or set LLAMACTL_HOST}}"
REMOTE_DIR="${2:-llama_controller}"
VENV="${VENV:-\$HOME/llama_controller_venv}"

cd "$(dirname "$0")"

echo "→ packing"
COPYFILE_DISABLE=1 tar czf - \
  --exclude='__pycache__' --exclude='.git' --exclude='config.json' --exclude='.DS_Store' \
  llamactl web requirements.txt deploy.sh README.md \
| ssh "$HOST" "rm -rf ~/$REMOTE_DIR/llamactl ~/$REMOTE_DIR/web \
    && mkdir -p ~/$REMOTE_DIR && tar xzf - -C ~/$REMOTE_DIR"

echo "→ deployed to $HOST:~/$REMOTE_DIR"
echo "   start with: ssh $HOST '$VENV/bin/python -m llamactl'  (cwd ~/$REMOTE_DIR)"

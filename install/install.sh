#!/usr/bin/env bash
# contour installer (macOS/Linux, secondary/best-effort — Windows install.ps1 is primary).
# Note: winctx.py's UIA text source is Windows-only; on other platforms the watcher
# falls back to the vision path for all captures.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES_DIR="${HOME}/.hermes"
HERMES_CONFIG="${HERMES_DIR}/config.yaml"
DATA_DIR="${HERMES_DIR}/contour"

RELAY_URL="${CONTOUR_RELAY_URL:-}"
DEVICE_TOKEN="${CONTOUR_DEVICE_TOKEN:-}"
ASK_CLOUD="${CONTOUR_ASK_CLOUD:-false}"
VISION_MODEL="${CONTOUR_VISION_MODEL:-gemma4:e4b}"
EMBED_MODEL="${CONTOUR_EMBED_MODEL:-nomic-embed-text}"

step() { printf "\n[%s] %s\n" "$1" "$2"; }

step 1 "Checking Python 3.10+"
command -v python3 >/dev/null || { echo "Python 3.10+ required"; exit 1; }
PYTHON="$(command -v python3)"

step 2 "Ensuring Ollama + models"
if command -v ollama >/dev/null; then
  ollama pull "$VISION_MODEL"; ollama pull "$EMBED_MODEL"
else
  echo "  ! Ollama not found — install from https://ollama.com/download"
fi

step 3 "Installing contour dependencies"
if command -v uv >/dev/null; then (cd "$PROJECT_ROOT" && uv sync); else "$PYTHON" -m pip install --user -e "$PROJECT_ROOT"; fi
"$PYTHON" -m pip install --user --quiet pyyaml

step 4 "Verifying Hermes"
command -v hermes >/dev/null && hermes doctor >/dev/null 2>&1 || echo "  ! Install/repair Hermes first"

step 5 "Registering MCP server + skill"
mkdir -p "$DATA_DIR"
"$PYTHON" "${PROJECT_ROOT}/install/register_hermes.py" \
  --config "$HERMES_CONFIG" --python "$PYTHON" \
  --relay-url "$RELAY_URL" --token "$DEVICE_TOKEN" \
  --data-dir "$DATA_DIR" --ask-cloud "$ASK_CLOUD"
mkdir -p "${HERMES_DIR}/skills/contour-activity"
cp "${PROJECT_ROOT}/skill/SKILL.md" "${HERMES_DIR}/skills/contour-activity/"

step 6 "Start the watcher manually:"
echo "  (cd \"$PROJECT_ROOT\" && \"$PYTHON\" -m watcher.daemon &)"
echo "Then ask Hermes: \"what was I just doing?\" (/reload-mcp if Hermes is already running)"

#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found in PATH." >&2
  exit 1
fi

echo "Serving UrbanAgentBench from: $REPO_ROOT"
echo "URL (Landing):         http://127.0.0.1:${PORT}/visualization/index.html"
echo "URL (Urban-Map-Web):   http://127.0.0.1:${PORT}/visualization/urban-map-web.html"
echo "URL (Urban-Satellite): http://127.0.0.1:${PORT}/visualization/urban-satellite.html"
echo "Press Ctrl+C to stop."

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://127.0.0.1:${PORT}/visualization/index.html" >/dev/null 2>&1 || true
fi

python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$REPO_ROOT"

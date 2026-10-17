#!/usr/bin/env bash
# PostToolUse (Edit|Write) — SPEC §3.4. Formatea y lintea el .py editado; exit 2 si quedan errores.
set -uo pipefail
cd "$(dirname "$0")/.."
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

file="$(jq -r '.tool_input.file_path // empty')"
[[ "$file" == *.py && -f "$file" ]] || exit 0

"$UV" run --frozen -q ruff format -q "$file" >/dev/null 2>&1
if ! out="$("$UV" run --frozen -q ruff check --fix -q "$file" 2>&1)"; then
  echo "ruff check en $file:" >&2
  echo "$out" >&2
  exit 2
fi
exit 0

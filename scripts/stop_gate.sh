#!/usr/bin/env bash
# Stop hook — SPEC §3.5. No deja terminar el turno con `make check` roto.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ "$(jq -r '.stop_hook_active // false')" == "true" ]]; then
  exit 0
fi

if ! out="$(make check 2>&1)"; then
  {
    echo "make check falla: arreglalo antes de terminar el turno."
    echo "--- resumen ---"
    echo "$out" | grep -E "^(FAILED|ERROR)|error:|errors?,|[0-9]+ (failed|passed)|Would reformat|Found [0-9]+ error|^make: \*\*\*" | tail -n 30
  } >&2
  exit 2
fi
exit 0

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC="$SCRIPT_DIR/xiangshan_mul_sim.c"
OUT="$SCRIPT_DIR/xiangshan_mul_sim.out"

gcc -O2 -std=c11 -Wall -Wextra -I"$SCRIPT_DIR/../isa" "$SRC" -o "$OUT"

"$OUT" "$@"

#!/usr/bin/env bash
# Convert merged HF weights to GGUF (F16) and quantise.
#
#   BASE=runs/merged OUT=models bash scripts/convert_gguf.sh
#
# Needs llama.cpp's converter and quantiser. Point LLAMA_CPP at a llama.cpp checkout
# (or its build tree) and it will find both.
set -euo pipefail

BASE=${BASE:-runs/merged}              # merged HF directory (scripts/merge_lora.py output)
OUT=${OUT:-models}
QUANTS=${QUANTS:-"Q8_0 Q4_K_M"}
NAME=${NAME:-$(basename "$BASE")}
LLAMA_CPP=${LLAMA_CPP:-}
PY=${PY:-python3}

mkdir -p "$OUT"

if [ -z "$LLAMA_CPP" ]; then
  echo "Set LLAMA_CPP to a llama.cpp checkout (or build tree)." >&2
  echo "  e.g. LLAMA_CPP=~/llama.cpp bash scripts/convert_gguf.sh" >&2
  exit 1
fi

CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
[ -f "$CONVERT" ] || CONVERT="$(find "$LLAMA_CPP" -maxdepth 2 -name convert_hf_to_gguf.py | head -1)"
QUANTIZE="$(find "$LLAMA_CPP" -type f -name 'llama-quantize' | head -1)"
[ -f "$CONVERT" ]  || { echo "convert_hf_to_gguf.py not found under $LLAMA_CPP" >&2; exit 1; }
[ -x "$QUANTIZE" ] || { echo "llama-quantize not found under $LLAMA_CPP (build it: cmake --build build --target llama-quantize)" >&2; exit 1; }

F16="$OUT/$NAME-F16.gguf"
echo "▶ F16  $F16"
"$PY" "$CONVERT" "$BASE" --outfile "$F16" --outtype f16

for q in $QUANTS; do
  dst="$OUT/$NAME-$q.gguf"
  echo "▶ $q  $dst"
  LD_LIBRARY_PATH="$(dirname "$QUANTIZE"):${LD_LIBRARY_PATH:-}" "$QUANTIZE" "$F16" "$dst" "$q"
done

ls -la "$OUT"

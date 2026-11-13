#!/usr/bin/env bash
# Remaining-time protocol, in the order the user fixed: GLM -> Qwen -> PGTNet.
# (Sonnet runs separately through the Batches API.) Each step is resumable on its own.
set -uo pipefail
ROOT=/path/to/LLM_PPM
cd "$ROOT"
LOGF=$ROOT/chain_remaining.log
say () { echo "[chain] $(date '+%F %H:%M:%S') $*" | tee -a "$LOGF"; }

FROM="${FROM:-glm}"   # glm | qwen | pgtnet: where to start (earlier steps already done)
if [ "$FROM" = glm ]; then
say "=== 1. GLM-5.3-Flash ==="
bash run_remaining_llms.sh zai glm-5.3-flash >> "$LOGF" 2>&1 || say "GLM: script exited with an error"
fi
if [ "$FROM" != pgtnet ]; then

say "=== 2. Qwen3.5-9B (local llama-server) ==="
bash run_remaining_llms.sh llamacpp primary >> "$LOGF" 2>&1 || say "Qwen: script exited with an error"
fi

say "=== 3. PGTNet (remaining time + elapsed) ==="
PGTQ=/path/to/PGTNet/experiments_fts5050/queue_remaining.sh
if [ -x "$PGTQ" ] || [ -f "$PGTQ" ]; then
  bash "$PGTQ" >> "$LOGF" 2>&1 || say "PGTNet: queue exited with an error"
else
  say "PGTNet: $PGTQ not present yet, skipping"
fi
say "chain finished"

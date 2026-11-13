#!/usr/bin/env bash
# PGTNet FULL LOG under the remaining-time protocol, nine paper logs, five training seeds
# (42..46 = repetitions 0..4, one full table per seed). Train/val = every case of the
# original log that is not a fixed-set test case; test = the fixed sets' test cases at
# their own prefix length; target = remaining time; the scorer adds the elapsed time back
# (eval_pgtnet_fts5050.py -> results_fts5050_remaining.csv).
# Waits for the GPU chain (Qwen + few-shot seeds) to end. Resumable per (log, seed).
set -uo pipefail
PGT=/path/to/PGTNet
EXP=$PGT/experiments_fts5050
LOG=$EXP/queue_full_remaining.log
EVAL=/path/to/LLM_PPM/eval_pgtnet_fts5050.py
ORDER="gen_drifted gen_baseline bpic2020_rfp bpic2020_domestic bpic2020_prepaid \
bpic2020_international bpic2020_permit bpic2011 bpic2012"
say () { echo "[pgt-full] $(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

say "waiting for the GPU chain to finish (Qwen + PGTNet few-shot seeds 43-46)"
while ps -eo args= | grep -qE "[r]un_qwen_then_pgtnet.sh|[q]ueue_remaining_all.sh|[f]ewshot_pgtnet.py|[l]lama-server"; do sleep 120; done
say "GPU chain finished"
until grep -q "BUILD DONE" "$EXP/build_full_remaining.log" 2>/dev/null; do say "waiting for the full-log splits"; sleep 300; done

for SEED in 42 43 44 45 46; do
  for S in $ORDER; do
    SHORT=${S}_full_remaining
    DIR=$EXP/$SHORT; [ "$SEED" != 42 ] && DIR=${DIR}_seed$SEED
    if [ -f "$DIR/predictions_raw.csv" ]; then
      say "=== $SHORT seed $SEED already computed, scoring only ==="
    else
      say "=== $SHORT seed $SEED ==="
      MEM_CAP=45G bash "$EXP/run_fts5050.sh" "$SHORT" "$SEED" >> "$LOG" 2>&1 || say "$SHORT seed $SEED FAILED"
    fi
    python3 "$EVAL" "$(basename "$DIR")" 2>&1 | tee -a "$LOG"
  done
done
say "all done"

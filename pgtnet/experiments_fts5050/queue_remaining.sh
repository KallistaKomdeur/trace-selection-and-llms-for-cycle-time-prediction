#!/usr/bin/env bash
# Few-shot PGTNet under the remaining-time protocol: the model is trained on its
# original target (time left after the prefix) and scored on prediction + elapsed,
# i.e. the total cycle time, like every other row. Same fixed sets, same test cases,
# same sampling as the total-time campaigns (BPI12 at 10%, everything else in full).
# Each log is resumable: rerunning the queue picks up where a killed run stopped.
set -uo pipefail
PGT=/path/to/PGTNet
PY=python
EVAL=/path/to/LLM_PPM/eval_fewshot_pgtnet.py
LOG=$PGT/experiments_fts5050/queue_remaining.log
ORDER="gen_drifted gen_baseline bpic2020_rfp bpic2020_domestic bpic2020_prepaid \
bpic2020_international bpic2020_permit bpic2012 bpic2011"
say () { echo "[pgt-remaining] $(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

for SETS in temporal controlflow; do
  [ "$SETS" = temporal ] && SUB=fewshot_remaining || SUB=fewshot_cf_remaining
  for SHORT in $ORDER; do
    FRAC=1.0; [ "$SHORT" = bpic2012 ] && FRAC=0.10
    say "=== $SHORT / $SETS (sample_frac $FRAC) ==="
    $PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log "$SHORT" --sets "$SETS" \
        --target remaining_time --max_epoch 100 --sample_frac "$FRAC" \
        2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
    python3 "$EVAL" --base="$PGT/experiments_fts5050/$SUB" "$SHORT" 2>&1 | tee -a "$LOG"
  done
done
say "all done"

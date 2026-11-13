#!/usr/bin/env bash
# Few-shot PGTNet, remaining-time protocol, three retrievals x five seeds.
#   sets:  temporal (lambda 0.5, 50/50 sets), lambda0 (temporal only), controlflow (lambda 1)
#   seeds: 42 (the first pass, dirs fewshot[_l0|_cf]_remaining/), then 43 44 45 46
#          (dirs ..._seed<N>/). Same test cases and same 10% BPI12 sample for every seed.
# Waits for queue_remaining.sh (seed 42, temporal + controlflow) to end, then fills in
# seed 42 lambda0 and the four other seeds, one full table at a time. Resumable per log.
set -uo pipefail
PGT=/path/to/PGTNet
PY=python
EVAL=/path/to/LLM_PPM/eval_fewshot_pgtnet.py
LOG=$PGT/experiments_fts5050/queue_remaining_all.log
ORDER="gen_drifted gen_baseline bpic2020_rfp bpic2020_domestic bpic2020_prepaid \
bpic2020_international bpic2020_permit bpic2012 bpic2011"
say () { echo "[pgt-all] $(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

run_one () {   # SETS SEED SHORT
  local SETS=$1 SEED=$2 SHORT=$3 SUB FRAC=1.0
  case $SETS in temporal) SUB=fewshot_remaining;; lambda0) SUB=fewshot_l0_remaining;; controlflow) SUB=fewshot_cf_remaining;; esac
  [ "$SEED" != 42 ] && SUB=${SUB}_seed${SEED}
  [ "$SHORT" = bpic2012 ] && FRAC=0.10
  say "=== $SHORT / $SETS / seed $SEED (sample_frac $FRAC) ==="
  $PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log "$SHORT" --sets "$SETS" --seed "$SEED" \
      --target remaining_time --max_epoch 100 --sample_frac "$FRAC" \
      2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
  python3 "$EVAL" --base="$PGT/experiments_fts5050/$SUB" "$SHORT" 2>&1 | tee -a "$LOG"
}

say "waiting for queue_remaining.sh to finish (seed 42: temporal + controlflow)"
while pgrep -f "queue_remaining.sh" >/dev/null; do sleep 120; done
say "queue_remaining.sh finished"

# lambda0 sets must exist (built by LLM_PPM/build_lambda0_sets.py)
for SHORT in $ORDER; do
  SRC=$($PY -c "import json;print([l['src'] for l in json.load(open('$PGT/experiments_fts5050/logs.json')) if l['short']=='$SHORT'][0])")
  [ -f "/path/to/LLM_PPM/logs/$SRC/${SRC}_lambda0_fixed_sets.json" ] || say "WARNING: lambda0 sets missing for $SHORT"
done

for SHORT in $ORDER; do run_one lambda0 42 "$SHORT"; done
for SEED in 43 44 45 46; do
  for SETS in temporal lambda0 controlflow; do
    for SHORT in $ORDER; do run_one "$SETS" "$SEED" "$SHORT"; done
  done
done
say "all done"

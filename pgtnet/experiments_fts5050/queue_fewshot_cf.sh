#!/usr/bin/env bash
# Few-shot PGTNet on the original control-flow-only sets, examples left truncated as
# they were generated. Same eight logs as the paper table, lightest first.
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
PY=/home/padela/anaconda3/envs/graphgps/bin/python
LOG=$PGT/experiments_fts5050/queue_fewshot_cf.log
ORDER="bpic2020_prepaid gen_drifted gen_baseline bpic2020_rfp bpic2020_domestic \
bpic2020_international bpic2020_permit bpic2011"

for SHORT in $ORDER; do
  echo "================================================" | tee -a "$LOG"
  echo "[cf] $(date) starting: $SHORT" | tee -a "$LOG"
  $PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log "$SHORT" --sets controlflow \
      --max_epoch 100 2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
  python3 /home/padela/Scrivania/LLM_PPM/eval_fewshot_pgtnet.py \
      --base=$PGT/experiments_fts5050/fewshot_cf "$SHORT" 2>&1 | tee -a "$LOG"
done
echo "[cf] all done $(date)" | tee -a "$LOG"

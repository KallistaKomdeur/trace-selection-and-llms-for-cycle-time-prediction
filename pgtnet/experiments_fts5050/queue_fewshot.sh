#!/usr/bin/env bash
# Few-shot PGTNet on every log, smallest graph load first. Each log is resumable:
# rerunning the queue picks up where a killed run stopped.
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
PY=/home/padela/anaconda3/envs/graphgps/bin/python
LOG=$PGT/experiments_fts5050/queue_fewshot.log
ORDER="helpdesk gen_drifted gen_baseline bpic2020_rfp bpic2020_domestic hospital_billing \
bpic2020_international bpic2020_permit bpic2015 bpic2012 bpic2011"

for SHORT in $ORDER; do
  echo "================================================" | tee -a "$LOG"
  echo "[fewshot] $(date) starting: $SHORT" | tee -a "$LOG"
  $PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log "$SHORT" --max_epoch 100 \
      2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
  python3 /home/padela/Scrivania/LLM_PPM/eval_fewshot_pgtnet.py "$SHORT" 2>&1 | tee -a "$LOG"
done
echo "[fewshot] all done $(date)" | tee -a "$LOG"

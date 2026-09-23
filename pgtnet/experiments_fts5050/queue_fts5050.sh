#!/usr/bin/env bash
# Sequential queue for the 50/50 fixed-sets campaign, smallest graph dataset first.
# Idempotent: a log whose predictions_matched.csv already exists is skipped.
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
LOG=$PGT/experiments_fts5050/queue_fts5050.log
ORDER="bpic2020_prepaid helpdesk gen_drifted gen_baseline bpic2020_rfp bpic2020_domestic \
hospital_billing bpic2015 bpic2020_international bpic2020_permit bpic2012 bpic2011"

for SHORT in $ORDER; do
  if [ -f "$PGT/experiments_fts5050/$SHORT/predictions_matched.csv" ]; then
    echo "[queue] $SHORT already done, skipping" | tee -a "$LOG"; continue
  fi
  echo "================================================" | tee -a "$LOG"
  echo "[queue] $(date) starting: $SHORT" | tee -a "$LOG"
  echo "================================================" | tee -a "$LOG"
  MEM_CAP="${MEM_CAP:-40G}" bash "$PGT/experiments_fts5050/run_fts5050.sh" "$SHORT" >> "$LOG" 2>&1 \
    || echo "[queue] $SHORT FAILED" | tee -a "$LOG"
  if [ -f "$PGT/experiments_fts5050/$SHORT/predictions_matched.csv" ]; then
    python3 /home/padela/Scrivania/LLM_PPM/eval_pgtnet_fts5050.py "$SHORT" 2>&1 | tee -a "$LOG"
  fi
done
echo "[queue] all done $(date)" | tee -a "$LOG"

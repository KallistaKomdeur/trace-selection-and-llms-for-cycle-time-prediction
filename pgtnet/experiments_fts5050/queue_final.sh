#!/usr/bin/env bash
# Celle mancanti che si possono ancora produrre in locale.
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
PY=/home/padela/anaconda3/envs/graphgps/bin/python
LOG=$PGT/experiments_fts5050/queue_final.log
say () { echo "[fin] $(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

say "=== PGTNet few-shot lambda=0.5 su bpic2012 (10% del test set) ==="
$PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log bpic2012 --sets temporal \
    --max_epoch 100 --sample_frac 0.10 2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
python3 /home/padela/Scrivania/LLM_PPM/eval_fewshot_pgtnet.py bpic2012 2>&1 | tee -a "$LOG"

say "=== PGTNet FULL LOG su bpic2011 ==="
MEM_CAP=45G bash "$PGT/experiments_fts5050/run_fts5050.sh" bpic2011_full >> "$LOG" 2>&1 \
  || say "bpic2011_full FALLITO"
python3 /home/padela/Scrivania/LLM_PPM/eval_pgtnet_fts5050.py bpic2011_full 2>&1 | tee -a "$LOG"
say "finito"

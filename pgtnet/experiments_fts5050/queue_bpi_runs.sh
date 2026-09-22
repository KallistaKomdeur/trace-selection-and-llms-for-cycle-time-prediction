#!/usr/bin/env bash
# Remaining PGTNet runs, serialised on the single GPU:
#   1. bpic2011 full training set        (run_fts5050.sh, explicit split)
#   2. bpi12w   few-shot control-flow    (one training per test trace)
#   3. bpi12w   full training set
# The few-shot bpic2011 on the temporal sets is already running and is waited on first.
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
PY=/home/padela/anaconda3/envs/graphgps/bin/python
LOG=$PGT/experiments_fts5050/queue_bpi_runs.log

say () { echo "[bpi] $(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

say "attendo il few-shot bpic2011 in corso"
while ps -eo args= | grep -q "[f]ewshot_pgtnet.py --log bpic2011 --sets temporal"; do sleep 60; done
python3 /home/padela/Scrivania/LLM_PPM/eval_fewshot_pgtnet.py bpic2011 2>&1 | tee -a "$LOG"

say "=== bpic2011 full training set ==="
MEM_CAP=45G bash "$PGT/experiments_fts5050/run_fts5050.sh" bpic2011_full >> "$LOG" 2>&1 \
  || say "bpic2011_full FALLITO"
python3 /home/padela/Scrivania/LLM_PPM/eval_pgtnet_fts5050.py bpic2011_full 2>&1 | tee -a "$LOG"

say "=== bpi12w few-shot control-flow ==="
$PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log bpi12w --sets controlflow --max_epoch 100 \
    2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
python3 /home/padela/Scrivania/LLM_PPM/eval_fewshot_pgtnet.py \
    --base=$PGT/experiments_fts5050/fewshot_cf bpi12w 2>&1 | tee -a "$LOG"

say "=== bpi12w full training set ==="
MEM_CAP=45G bash "$PGT/experiments_fts5050/run_fts5050.sh" bpi12w_full >> "$LOG" 2>&1 \
  || say "bpi12w_full FALLITO"
python3 /home/padela/Scrivania/LLM_PPM/eval_pgtnet_fts5050.py bpi12w_full 2>&1 | tee -a "$LOG"

say "tutto finito"

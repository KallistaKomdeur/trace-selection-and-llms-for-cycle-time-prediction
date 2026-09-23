#!/usr/bin/env bash
# helpdesk: PGTNet few-shot lambda=1 sul 10% del test set, poi PGTNet sul training set completo
# (il test completo li' non costa nulla in piu': il costo e' il training).
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet; ROOT=/home/padela/Scrivania/LLM_PPM
PY=/home/padela/anaconda3/envs/graphgps/bin/python
LOG=$PGT/experiments_fts5050/queue_helpdesk.log
say () { echo "[hd] $(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }
say "=== PGTNet few-shot lambda=1 helpdesk (10%) ==="
$PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log helpdesk --sets controlflow --max_epoch 100 --sample_frac 0.10 \
    2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
python3 "$ROOT/eval_fewshot_pgtnet.py" --base=$PGT/experiments_fts5050/fewshot_cf helpdesk 2>&1 | tee -a "$LOG"
say "=== PGTNet FULL LOG helpdesk ==="
MEM_CAP=45G bash "$PGT/experiments_fts5050/run_fts5050.sh" helpdesk_full >> "$LOG" 2>&1 || say "helpdesk_full FALLITO"
$PY "$ROOT/eval_pgtnet_fts5050.py" helpdesk_full 2>&1 | grep -v Warning | tee -a "$LOG"
say "finito"

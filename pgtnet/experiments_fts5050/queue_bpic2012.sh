#!/usr/bin/env bash
# Few-shot temporal on bpic2012, after the four bpi runs release the GPU.
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
PY=/home/padela/anaconda3/envs/graphgps/bin/python
LOG=$PGT/experiments_fts5050/queue_bpi_runs.log
while ps -eo args= | grep -q "[q]ueue_bpi_runs.sh"; do sleep 120; done
echo "[bpi] $(date '+%F %H:%M:%S') === bpic2012 few-shot temporale ===" | tee -a "$LOG"
$PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log bpic2012 --sets temporal --max_epoch 100 \
    2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
python3 /home/padela/Scrivania/LLM_PPM/eval_fewshot_pgtnet.py bpic2012 2>&1 | tee -a "$LOG"
echo "[bpi] $(date '+%F %H:%M:%S') bpic2012 finito" | tee -a "$LOG"

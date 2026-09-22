#!/usr/bin/env bash
# Tutte le celle mancanti ancora producibili, dopo la coda queue_final in corso.
# PGTNet prima (GPU esclusiva), Qwen in fondo con un solo avvio di llama-server.
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
ROOT=/home/padela/Scrivania/LLM_PPM
PY=/home/padela/anaconda3/envs/graphgps/bin/python
LOG=$PGT/experiments_fts5050/queue_gaps.log
say () { echo "[gap] $(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

say "attendo queue_final"
while ps -eo args= | grep -q "[q]ueue_final.sh"; do sleep 120; done

# --- PGTNet few-shot lambda=1 sui due log nuovi, 10% del test set
for L in bpic2012 hospital_billing; do
  say "=== PGTNet few-shot lambda=1 $L (10%) ==="
  $PY "$PGT/experiments_fts5050/fewshot_pgtnet.py" --log "$L" --sets controlflow \
      --max_epoch 100 --sample_frac 0.10 2>&1 | grep -viE "it/s\]|^Processing|^Done" >> "$LOG"
  python3 "$ROOT/eval_fewshot_pgtnet.py" --base=$PGT/experiments_fts5050/fewshot_cf "$L" \
      2>&1 | tee -a "$LOG"
done

# --- PGTNet allenato su tutto il log
for L in bpic2012_full hospital_billing_full; do
  say "=== PGTNet FULL LOG $L ==="
  MEM_CAP=45G bash "$PGT/experiments_fts5050/run_fts5050.sh" "$L" >> "$LOG" 2>&1 \
    || say "$L FALLITO"
  python3 "$ROOT/eval_pgtnet_fts5050.py" "$L" 2>&1 | tee -a "$LOG"
done

# --- Qwen: lambda=1 e Random sui due log nuovi
say "avvio llama-server"
/home/padela/llama.cpp-cuda/build/bin/llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M \
    -ngl 99 -t 8 -c 16384 --jinja --port 8080 > "$ROOT/llama_server_gaps.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf --max-time 3 http://127.0.0.1:8080/props >/dev/null && break; sleep 5; done
curl -sf --max-time 3 http://127.0.0.1:8080/props >/dev/null || { say "llama-server KO"; exit 2; }
say "llama-server pronto"
for MODE in similar_prefix random; do
  SET=$ROOT/config/settings_${MODE}.yaml
  sed "s/^selection_mode:.*/selection_mode: ${MODE}/" "$ROOT/config/settings.yaml" > "$SET"
  for L in bpic2012 hospital_billing; do
    say "=== Qwen $MODE $L ==="
    PPM_SETTINGS="$SET" PPM_RESULTS_DIR="$ROOT/results_qwen_${MODE}" \
      python3 "$ROOT/test_llm.py" "$L" llamacpp single_ref --model primary >> "$LOG" 2>&1 \
      || say "Qwen $MODE $L FALLITO"
  done
done
say "tutto finito"

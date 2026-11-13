#!/usr/bin/env bash
# The remaining-time protocol (prompts/single_ref_remaining.txt) for the models that
# cost nothing or little: the LLM is asked for the remaining time of the running case
# and the scored prediction is that answer plus the elapsed time at the prefix.
# Nine paper logs x both selection strategies, same fixed sets as every other row.
#
#   bash run_remaining_llms.sh llamacpp primary        # Qwen3.5-9B, local llama-server
#   ZAI_API_KEY=... bash run_remaining_llms.sh zai glm-5.3-flash
#
# Resumable: test_llm.py skips generations already in the results tree.
set -uo pipefail
ROOT=/path/to/LLM_PPM
cd "$ROOT"
PROVIDER="${1:?provider (llamacpp|zai)}"
MODEL="${2:?model key (primary|glm-5.3-flash)}"
CONFIG=single_ref_remaining
TAG=$PROVIDER
[ "$PROVIDER" = "llamacpp" ] && TAG=qwen
[ "$PROVIDER" = "zai" ] && TAG=glm
REP="${REP:-0}"                      # repetition index (0 = the first run)
export PPM_REPETITION=$REP
RSUF=""; [ "$REP" != 0 ] && RSUF="_rep$REP"
RES=$ROOT/results_${TAG}_${CONFIG}$RSUF
LOGF=$ROOT/${TAG}_${CONFIG}$RSUF.log
LLAMA=llama-server
# The CUDA build of llama-server links libcudart/libcublas 12, which on this machine
# live only inside the conda nvidia wheels; a non-login shell has no LD_LIBRARY_PATH.
NV=/path/to/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
LOGS="bpic2012 bpic2011 bpic2020_domestic_declarations bpic2020_international_declarations \
bpic2020_prepaid_travel_costs bpic2020_request_for_payment bpic2020_travel_permit_data \
gen_baseline gen_drifted"
say () { echo "[$TAG-remaining] $(date '+%H:%M:%S') $*" | tee -a "$LOGF"; }

SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null && say "llama-server stopped"' EXIT

if [ "$PROVIDER" = "zai" ]; then
  if [ -z "${ZAI_API_KEY:-}" ] && ! grep -q "^ZAI_API_KEY=." .env 2>/dev/null; then
    read -rsp "API key Z AI (input nascosto): " ZAI_API_KEY; echo
    [ -n "$ZAI_API_KEY" ] || { echo "No API key."; exit 1; }
    export ZAI_API_KEY
  fi
fi
server_up () { curl -sf --max-time 3 http://127.0.0.1:8080/props >/dev/null; }
# (Re)starts llama-server when it is not answering. Called before every log: on
# 2026-11-04 the server segfaulted mid-campaign and every later log failed in a
# second, so a single start at the beginning is not enough.
ensure_server () {
  [ "$PROVIDER" = "llamacpp" ] || return 0
  server_up && return 0
  [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
  say "starting llama-server"
  # --no-mmproj: the vision projector (880 MiB on the GPU) is never used by text prompts
  # and twice pushed the 8 GB card into CUDA out-of-memory (2026-11-04/05).
  "$LLAMA" -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M --no-mmproj -ngl 99 -t 8 -c 16384 --jinja --port 8080 \
      >> "$ROOT/llama_server_remaining.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do server_up && break; sleep 5; done
  server_up || { say "llama-server not responding"; return 2; }
  say "llama-server ready"
}
ensure_server || exit 2

export PPM_RESULTS_DIR=$RES
FAILED=""
# similar_prefix_temporal = lambda 0.5 (50/50 sets), similar_prefix = lambda 1
# (control-flow only), random = no selection. MODES can be overridden from the env.
MODES="${MODES:-similar_prefix_temporal random similar_prefix lambda0}"
for MODE in $MODES; do
  case $MODE in
    random)         export PPM_SETTINGS=$ROOT/config/settings_random.yaml ;;
    similar_prefix) export PPM_SETTINGS=$ROOT/config/settings_similar_prefix.yaml ;;
    lambda0)        export PPM_SETTINGS=$ROOT/config/settings_lambda0.yaml ;;
    *)              export PPM_SETTINGS=$ROOT/config/settings_fts5050.yaml ;;
  esac
  for L in $LOGS; do
    say "=== $L / $MODE ==="
    # Two attempts per log: a server crash mid-log ends test_llm.py with an error, the
    # server is restarted and the log resumes from the cases already saved.
    OK=0
    for ATTEMPT in 1 2; do
      ensure_server || break
      python3 test_llm.py "$L" "$PROVIDER" "$CONFIG" --model "$MODEL" >> "$LOGF" 2>&1 && { OK=1; break; }
      say "$L/$MODE attempt $ATTEMPT failed"
    done
    [ "$OK" = 1 ] || { say "$L/$MODE FAILED"; FAILED="$FAILED $L/$MODE"; }
  done
done
say "campaign finished.${FAILED:+ failed:$FAILED}"
for MODE in $MODES; do
  python3 eval_llm_fts5050.py --provider "$PROVIDER" --model "$MODEL" --mode "$MODE" \
    --configuration "$CONFIG" --repetition "$REP" --results-dir "$RES" --logs $LOGS 2>&1 | tee -a "$LOGF"
done

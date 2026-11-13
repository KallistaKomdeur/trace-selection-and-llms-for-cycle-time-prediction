#!/usr/bin/env bash
# Second Anthropic key (ANTHROPIC_API_KEY_2 in .env), budget 100 USD at batch rates.
# 1. wait for the lambda=1 campaign running on the first key to end;
# 2. with key 2: resubmit whatever lambda=1 cases failed for lack of credit (resumable:
#    only missing generations are sent), then the lambda=0 campaign;
# 3. score both strategies and print the table rows.
set -uo pipefail
ROOT=/path/to/LLM_PPM
cd "$ROOT"
LOGF=$ROOT/sonnet55_key2.log
RES=$ROOT/results_sonnet55_single_ref_remaining
CONFIG=single_ref_remaining
MODEL=claude-sonnet-5-5
MAXUSD=100
say () { echo "[sonnet55-key2] $(date '+%F %H:%M:%S') $*" | tee -a "$LOGF"; }

say "waiting for the lambda=1 campaign on the first key"
while pgrep -f "run_anthropic_batch.py --configuration single_ref_remaining --mode similar_prefix " >/dev/null; do sleep 60; done
say "lambda=1 campaign (key 1) finished"

# key 2 from .env, exported in-process (never on a command line)
export ANTHROPIC_API_KEY="$(grep '^ANTHROPIC_API_KEY_2=' .env | cut -d= -f2-)"
[ -n "$ANTHROPIC_API_KEY" ] || { say "ANTHROPIC_API_KEY_2 missing in .env"; exit 1; }
python3 -m utils.anthropic_backend 2>&1 | tee -a "$LOGF" | grep -q '"generate": "ok' || { say "selftest with key 2 failed"; exit 1; }

ORDER="gen_baseline gen_drifted bpic2020_prepaid_travel_costs bpic2020_request_for_payment \
bpic2020_domestic_declarations bpic2020_international_declarations bpic2020_travel_permit_data bpic2011 bpic2012"

say "=== lambda=1: completing the missing cases (key 2) ==="
python3 run_anthropic_batch.py --configuration $CONFIG --mode similar_prefix --results-dir "$RES" \
  --max-usd $MAXUSD --logs $ORDER 2>&1 | tee -a "$LOGF"

say "=== lambda=0: full campaign (key 2, cap $MAXUSD USD) ==="
python3 run_anthropic_batch.py --configuration $CONFIG --mode lambda0 --results-dir "$RES" \
  --max-usd $MAXUSD --logs $ORDER 2>&1 | tee -a "$LOGF"

say "=== evaluation ==="
for M in similar_prefix lambda0; do
  python3 eval_llm_fts5050.py --provider anthropic --model $MODEL --mode $M --configuration $CONFIG \
    --results-dir "$RES" --logs $ORDER 2>&1 | tee -a "$LOGF"
done
python3 build_table_rows.py --configuration $CONFIG 2>/dev/null | grep -i sonnet | tee -a "$LOGF"
say "done"

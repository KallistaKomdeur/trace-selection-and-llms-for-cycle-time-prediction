#!/usr/bin/env bash
# Claude Sonnet 5.5 as the fourth LLM of the comparison, end to end.
#
#   1. selftest of the backend (key, model id, one tiny request)
#   2. cost estimate of the whole campaign at Batch API rates (exact token counts,
#      free endpoint); aborts if above --max-usd
#   3. latency sample: 30 synchronous predictions per log on the similar-prefix sets,
#      the per-request timing Table 1 of the paper reports (results_sonnet55_latency/)
#   4. the full campaign, both selection strategies, through the Batches API
#      (results_sonnet55/), resumable: re-run the script and it continues
#   5. scoring with the same metrics as every other row, and the LaTeX table
#
# The key comes from ANTHROPIC_API_KEY in .env (or the environment), never from an
# argument. Usage:  [CONFIG=single_ref_remaining] bash run_sonnet55.sh [max_usd]
#   CONFIG   prompt template: single_ref (paper, asks the total time) or
#            single_ref_remaining (asks the remaining time; llm_answer = answer + elapsed).
#            A non-default CONFIG gets its own results trees and CSVs.
set -uo pipefail
ROOT=/path/to/LLM_PPM
cd "$ROOT"
MAXUSD="${1:-99}"
CONFIG="${CONFIG:-single_ref}"
SUFFIX=""; [ "$CONFIG" = "single_ref" ] || SUFFIX="_${CONFIG}"
RES=$ROOT/results_sonnet55$SUFFIX
LAT=$ROOT/results_sonnet55_latency$SUFFIX
LOGF=$ROOT/sonnet55$SUFFIX.log
MODEL=claude-sonnet-5-5
say () { echo "[sonnet55] $(date '+%H:%M:%S') $*" | tee -a "$LOGF"; }

LOGS="bpic2012 bpic2011 bpic2020_domestic_declarations bpic2020_international_declarations \
bpic2020_prepaid_travel_costs bpic2020_request_for_payment bpic2020_travel_permit_data \
gen_baseline gen_drifted"

say "=== 1. selftest ==="
python3 -m utils.anthropic_backend 2>&1 | tee -a "$LOGF" | grep -q '"generate": "ok' \
  || { say "selftest failed: check ANTHROPIC_API_KEY in .env"; exit 1; }

say "configuration=$CONFIG  results=$RES  latency=$LAT  cap=$MAXUSD USD"
say "=== 2. cost estimate (batch rates) ==="
python3 run_anthropic_batch.py --configuration "$CONFIG" --results-dir "$RES" --estimate 2>&1 | tee -a "$LOGF"

say "=== 3. latency sample (30 cases/log, synchronous, similar_prefix_temporal) ==="
if [ -n "${SKIP_LATENCY:-}" ]; then
  say "skipped (SKIP_LATENCY): per-request latency does not depend on the prompt, results_sonnet55_latency/ is reused"
  LOGS_LAT=""
else
  LOGS_LAT=$LOGS
fi
export PPM_SETTINGS=$ROOT/config/settings_fts5050.yaml
export PPM_RESULTS_DIR=$LAT
for L in $LOGS_LAT; do
  say "latency $L"
  python3 test_llm.py "$L" anthropic "$CONFIG" --model $MODEL --pilot 30 >> "$LOGF" 2>&1 \
    || say "$L latency FAILED"
done
unset PPM_SETTINGS PPM_RESULTS_DIR

say "=== 4. full campaign via the Batch API ==="
python3 run_anthropic_batch.py --configuration "$CONFIG" --results-dir "$RES" --max-usd "$MAXUSD" 2>&1 \
  | tee -a "$LOGF" || { say "campaign interrupted"; exit 1; }

say "=== 5. evaluation ==="
for M in similar_prefix_temporal random; do
  python3 eval_llm_fts5050.py --provider anthropic --model $MODEL --mode $M --configuration "$CONFIG" \
    --results-dir "$RES" 2>&1 | tee -a "$LOGF"
done
python3 build_table_rows.py --configuration "$CONFIG" 2>/dev/null | grep -i "sonnet" | tee -a "$LOGF"
say "done. CSV: results_anthropic_${MODEL}_similar_prefix_temporal$SUFFIX.csv, results_anthropic_${MODEL}_random$SUFFIX.csv"

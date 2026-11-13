#!/usr/bin/env bash
# GLM-5.3-Flash, remaining-time protocol, repetitions 0..4 (0 = the first run, partly done)
# for lambda = 0.5, 0 and 1. Each (repetition, strategy) pass is resumable, and the queue
# pauses on "Insufficient balance" (Z AI code 1113) until a probe request is accepted
# again, so a recharge resumes it without any manual step.
set -uo pipefail
ROOT=/path/to/LLM_PPM
cd "$ROOT"
LOGF=$ROOT/glm_repetitions.log
say () { echo "[glm-reps] $(date '+%F %H:%M:%S') $*" | tee -a "$LOGF"; }

probe_ok () {   # one tiny request; true when the balance is active
  python3 - <<'PY' >/dev/null 2>&1
from dotenv import load_dotenv; load_dotenv(".env")
from utils.zai_backend import get_shared_zai_backend
get_shared_zai_backend("glm-5.3-flash").generate("Reply with the single digit 7 and nothing else.")
PY
}
wait_balance () {
  until probe_ok; do say "Z AI balance too low, retrying in 10 min"; sleep 600; done
}

# Repetition 0 is the first run: its lambda=0.5 is complete (skipped in seconds by the
# ledger), lambda=1 and lambda=0 are finished here, before the other repetitions.
for REP in 0 1 2 3 4; do
  for MODE in similar_prefix_temporal lambda0 similar_prefix; do
    for attempt in 1 2 3 4 5 6 7 8; do
      wait_balance
      say "=== repetition $REP / $MODE (attempt $attempt) ==="
      RLOG="$ROOT/glm_single_ref_remaining.log"; [ "$REP" != 0 ] && RLOG="$ROOT/glm_single_ref_remaining_rep$REP.log"
      MARK=$(wc -l < "$RLOG" 2>/dev/null || echo 0)
      REP=$REP MODES=$MODE bash run_remaining_llms.sh zai glm-5.3-flash >> "$LOGF" 2>&1
      if tail -n +$((MARK + 1)) "$RLOG" 2>/dev/null | grep -q "Insufficient balance"; then
        say "credit exhausted during repetition $REP / $MODE: resuming after top-up"
        continue
      fi
      break
    done
  done
done
say "all repetitions finished"

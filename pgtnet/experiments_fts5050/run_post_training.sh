#!/usr/bin/env bash
# Post-training pipeline for one experiment:
#   1. Run inference (event-inference mode) on the test set.
#   2. Run ResultHandler.py to attach case ids / prefix lengths.
#   3. Archive: train/val/test logs, model, predictions, real values into experiments/<short>/
#
# Usage: bash run_post_training.sh <short_name> <jsonl_in_new_logs> <dataset_class> <inference_cfg_basename>
# Example:
#   bash run_post_training.sh helpdesk helpdesk_preprocessed.jsonl HelpDeskV2 helpdeskv2-GPSwGraphormer-ckptbest-eventinference

set -euo pipefail

SHORT="$1"          # helpdesk
JSONL="$2"          # helpdesk_preprocessed.jsonl
DATASET="$3"        # HelpDeskV2 (matches PGTNetutils naming)
INF_CFG="$4"        # helpdeskv2-GPSwGraphormer-ckptbest-eventinference

PGT=/home/padela/Scrivania/PGTNet
GPS=$PGT/GraphGPS
EXP_DIR=$PGT/experiments/$SHORT
RAW_PICKLES=$GPS/datasets/EVENT${DATASET}/raw

mkdir -p "$EXP_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate graphgps

# ---- 1. Inference -----------------------------------------------------------
cd "$GPS"
python main.py --cfg configs/GPS/${INF_CFG}.yaml run_multiple_splits [0] seed 42 \
    > "$EXP_DIR/inference.log" 2>&1
echo "[done] inference"

# ---- 2. ResultHandler -------------------------------------------------------
cd "$PGT"
python ResultHandler.py --dataset_name "$DATASET" --seed_number 42 \
    --inference_config "$INF_CFG" > "$EXP_DIR/result_handler.log" 2>&1
echo "[done] result handler"

# ---- 3. Archive into experiment folder -------------------------------------
# Train / val / test pickle splits (the actual graph dataset used)
cp "$RAW_PICKLES"/{train,val,test}.pickle "$EXP_DIR/"

# Source log (audit trail). Try new_logs/, raw_dataset/, then new_data/.
EXT="${JSONL##*.}"
if   [ -f "$PGT/new_logs/$JSONL" ];    then cp "$PGT/new_logs/$JSONL"    "$EXP_DIR/source.${EXT}"
elif [ -f "$PGT/raw_dataset/$JSONL" ]; then cp "$PGT/raw_dataset/$JSONL" "$EXP_DIR/source.${EXT}"
elif [ -f "$PGT/new_data/$JSONL" ];    then cp "$PGT/new_data/$JSONL"    "$EXP_DIR/source.${EXT}"
else echo "[warn] source log $JSONL not found in new_logs/, raw_dataset/, or new_data/"
fi

# Trained model checkpoint
CKPT_DIR=$GPS/results/${INF_CFG%-eventinference}/0/ckpt
if [ -d "$CKPT_DIR" ]; then
  cp "$CKPT_DIR"/*.ckpt "$EXP_DIR/model.ckpt"
fi

# Raw prediction dataframe (the per-graph predictions)
PRED=$GPS/results/${INF_CFG}/${DATASET}-pgtnet_prediction_dataframe.csv
if [ -f "$PRED" ]; then
  cp "$PRED" "$EXP_DIR/predictions_raw.csv"
fi

# Matched dataframe (with cid + prefix length)
MATCHED=$PGT/"PGTNet results"/$DATASET/"seed 42"/${DATASET}-seed42-PGTNet_results.csv
if [ -f "$MATCHED" ]; then
  cp "$MATCHED" "$EXP_DIR/predictions_matched.csv"
fi

# Final report — extract real vs predicted (in days) into a tidy CSV
python - <<PY
import pandas as pd, os
from PGTNetutils import mean_cycle_norm_factor_provider
norm, mean_cycle = mean_cycle_norm_factor_provider("$DATASET")
exp = "$EXP_DIR"
src = os.path.join(exp, "predictions_raw.csv")
out = os.path.join(exp, "predictions_days.csv")
df = pd.read_csv(src)
df["real_days"]      = df["real_cycle_time"]      * norm
df["predicted_days"] = df["predicted_cycle_time"] * norm
df["abs_error_days"] = (df["real_days"] - df["predicted_days"]).abs()
df.to_csv(out, index=False)
mae = df["abs_error_days"].mean()
print(f"MAE (days): {mae:.4f}")
print(f"Relative MAE (%): {mae / mean_cycle * 100:.2f}")
print(f"Rows: {len(df)}")
with open(os.path.join(exp, "RESULTS.md"), "w") as f:
    f.write(f"# {os.path.basename(exp)} — PGTNet results\n\n")
    f.write(f"- Dataset class:  EVENT{ '$DATASET' }\n")
    f.write(f"- Test prefixes:  {len(df)}\n")
    f.write(f"- Normalization:  {norm:.4f} days\n")
    f.write(f"- Mean cycle:     {mean_cycle:.2f} days\n")
    f.write(f"- **MAE (days):** {mae:.4f}\n")
    f.write(f"- **Relative MAE:** {mae/mean_cycle*100:.2f}%\n")
PY
echo "[done] archived to $EXP_DIR"
ls -la "$EXP_DIR"

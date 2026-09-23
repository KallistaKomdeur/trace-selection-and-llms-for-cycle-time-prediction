#!/usr/bin/env bash
# Launch one PGTNet training run with num_workers=12 for the data loader,
# wait for it, then run inference + ResultHandler + archive.
#
# Usage: bash launch_train.sh <short> <jsonl_filename> <DatasetClass> <cfg_basename>
# Example:
#   bash launch_train.sh bpic2020_rfp bpic2020_request_for_payment_preprocessed.jsonl BPIC20RFP \
#                        bpic2020rfp-GPSwGraphormer-ckptbest

set -euo pipefail
SHORT="$1"
JSONL="$2"
DATASET="$3"
CFG_BASE="$4"            # e.g. bpic2020rfp-GPSwGraphormer-ckptbest

PGT=/home/padela/Scrivania/PGTNet
GPS=$PGT/GraphGPS
EXP=$PGT/experiments/$SHORT
mkdir -p "$EXP"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate graphgps

# Re-sync scripts/configs to GraphGPS clone (covers any edge encoder updates)
cp $PGT/scripts/master_loader.py            $GPS/graphgps/loader/master_loader.py
cp $PGT/scripts/GTeventlogHandler.py        $GPS/graphgps/loader/dataset/GTeventlogHandler.py
cp $PGT/scripts/linear_edge_encoder.py      $GPS/graphgps/encoder/linear_edge_encoder.py
cp $PGT/scripts/two_layer_linear_edge_encoder.py $GPS/graphgps/encoder/two_layer_linear_edge_encoder.py
cp $PGT/training_configs/${CFG_BASE}.yaml $GPS/configs/GPS/
cp $PGT/inference_configs/${CFG_BASE}-eventinference.yaml $GPS/configs/GPS/

cd "$GPS"

# Hard-cap RAM via a transient systemd scope; OOM-kill cleanly if exceeded.
MEM_CAP="${MEM_CAP:-20G}"
NUM_WORKERS="${NUM_WORKERS:-12}"
systemd-run --user --scope --quiet \
    -p MemoryMax=${MEM_CAP} \
    -p MemorySwapMax=0 \
    -- python main.py --cfg configs/GPS/${CFG_BASE}.yaml \
       run_multiple_splits [0] seed 42 num_workers ${NUM_WORKERS} \
    > "$EXP/train.log" 2>&1 || echo "[train returned non-zero, continuing]"
echo "[train done] $SHORT"

# Inference + ResultHandler + archive — reuse run_post_training.sh
bash $PGT/experiments/run_post_training.sh \
    "$SHORT" "$JSONL" "$DATASET" "${CFG_BASE}-eventinference" \
    || echo "[post_training returned non-zero, continuing]"

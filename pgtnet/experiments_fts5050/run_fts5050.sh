#!/usr/bin/env bash
# One PGTNet run on the 50/50 similar-prefix fixed sets.
#
# Everything the model sees comes from similar_prefix_temporal_fixed_sets_5050/:
# train/val are the cases listed under "examples", test is exactly the fixed sets'
# test cases at exactly their prefix length, with targets from "true_total_time".
# No case is selected by a ratio -- fts5050_splits/<short>.json drives the split.
#
# Existing dataset classes / training configs / inference configs are reused, so
# every artefact this touches is backed up and restored by the EXIT trap.
#
# Usage: bash run_fts5050.sh <short>
set -uo pipefail
SHORT="$1"
PGT=/home/padela/Scrivania/PGTNet
GPS=$PGT/GraphGPS
PY=/home/padela/anaconda3/envs/graphgps/bin/python
TS=$(date +%s)

read -r DATASET XES CFG < <($PY - "$SHORT" <<'EOF'
import json, sys
for l in json.load(open('/home/padela/Scrivania/PGTNet/experiments_fts5050/logs.json')):
    if l['short'] == sys.argv[1]:
        print(l['dataset'], l['xes'], l['cfg']); break
else:
    sys.exit(f"unknown short {sys.argv[1]}")
EOF
)
[ -n "${DATASET:-}" ] || { echo "[$SHORT] cannot resolve dataset"; exit 2; }

JSONL=${SHORT}_fts5050_preprocessed.jsonl
EXP_FULL=$PGT/experiments/$SHORT
EXP_OUT=$PGT/experiments_fts5050/$SHORT
GPS_DS=$GPS/datasets/EVENT${DATASET}
ROGUE=/home/padela/Scrivania/datasets/EVENT${DATASET}
XES_PATH=$PGT/raw_dataset/$XES

restore () {
  echo "[$SHORT] restore..."
  for d in raw processed; do
    if [ -d "$GPS_DS/$d.bak_$TS" ]; then rm -rf "$GPS_DS/$d"; mv "$GPS_DS/$d.bak_$TS" "$GPS_DS/$d"; fi
  done
  [ -f "$XES_PATH.bak_$TS" ] && { rm -f "$XES_PATH"; mv "$XES_PATH.bak_$TS" "$XES_PATH"; }
  for enc in linear_edge_encoder two_layer_linear_edge_encoder; do
    [ -f "$PGT/scripts/${enc}.py.bak_$TS" ] && mv "$PGT/scripts/${enc}.py.bak_$TS" "$PGT/scripts/${enc}.py"
  done
  if [ -d "$PGT/experiments/${SHORT}.bak_$TS" ]; then
    rm -rf "$EXP_FULL"; mv "$PGT/experiments/${SHORT}.bak_$TS" "$EXP_FULL"
  fi
  rm -rf "$ROGUE" 2>/dev/null
}
trap restore EXIT

mkdir -p "$EXP_OUT"
echo "[$SHORT] dataset=$DATASET xes=$XES cfg=$CFG"

# 1. XES from the fixed-sets JSONL (backing up the log's real XES first)
[ -f "$XES_PATH" ] && cp "$XES_PATH" "$XES_PATH.bak_$TS"
cd "$PGT"
$PY jsonl_to_xes.py --jsonl "$JSONL" > "$EXP_OUT/jsonl_to_xes.log" 2>&1 \
  || { echo "[$SHORT] jsonl_to_xes FAILED"; tail -5 "$EXP_OUT/jsonl_to_xes.log"; exit 3; }

# 2. Graph conversion (GTconvertor writes to a hardcoded parent-of-cwd path)
for d in raw processed; do [ -d "$GPS_DS/$d" ] && mv "$GPS_DS/$d" "$GPS_DS/$d.bak_$TS"; done
mkdir -p "$GPS_DS/raw" "$GPS_DS/processed"
rm -rf "$ROGUE"
$PY GTconvertor.py conversion_configs "${SHORT}_fts5050.yaml" --overwrite True \
    > "$EXP_OUT/convert.log" 2>&1
if [ ! -f "$ROGUE/raw/train.pickle" ]; then
  echo "[$SHORT] conversion FAILED"; tail -20 "$EXP_OUT/convert.log"; exit 4
fi
cp "$ROGUE/raw/"{train,val,test}.pickle "$GPS_DS/raw/"
rm -rf "$ROGUE"

# 3. Edge feature dim is data-dependent; the encoders hardcode it per cfg.dataset.name
NEW_DIM=$($PY -c "
import pickle
print(pickle.load(open('$GPS_DS/raw/train.pickle','rb'))[0].edge_attr.shape[1])")
DS_NAME=$($PY -c "
import yaml; print(yaml.safe_load(open('$PGT/training_configs/${CFG}.yaml'))['dataset']['name'])")
echo "[$SHORT] edge dim=$NEW_DIM  cfg.dataset.name=$DS_NAME"
for enc in linear_edge_encoder two_layer_linear_edge_encoder; do
  cp "$PGT/scripts/${enc}.py" "$PGT/scripts/${enc}.py.bak_$TS"
  $PY - <<PY
import re, sys
p = "$PGT/scripts/${enc}.py"
src = open(p).read()
pat = r"(elif cfg\.dataset\.name == '${DS_NAME}':\s*\n\s*self\.in_dim = )(\d+)"
m = re.search(pat, src)
if not m:
    sys.exit("FATAL: no in_dim branch for ${DS_NAME} in " + p)
# A no-op sub is fine and common: the stored dim already matches the data.
print(("unchanged (already ${NEW_DIM})" if m.group(2) == "${NEW_DIM}"
       else "patched %s -> ${NEW_DIM}" % m.group(2)), p)
open(p, "w").write(re.sub(pat, r"\g<1>${NEW_DIM}", src, count=1))
PY
done

# 4. Train + inference + archive
[ -d "$EXP_FULL" ] && mv "$EXP_FULL" "$PGT/experiments/${SHORT}.bak_$TS"
MEM_CAP="${MEM_CAP:-30G}" NUM_WORKERS="${NUM_WORKERS:-0}" \
  bash "$PGT/experiments/launch_train.sh" "$SHORT" "$JSONL" "$DATASET" "$CFG" \
  > "$EXP_OUT/run.log" 2>&1
echo "[$SHORT] launch_train returned $?"

if [ -d "$EXP_FULL" ]; then
  for f in "$EXP_FULL"/*; do mv "$f" "$EXP_OUT/" 2>/dev/null || true; done
  rmdir "$EXP_FULL" 2>/dev/null || true
fi
cp "$PGT/fts5050_splits/${SHORT}.json" "$EXP_OUT/split.json"
grep -E "max_time_norm|explicit split" "$EXP_OUT/convert.log" > "$EXP_OUT/norm.txt" 2>/dev/null
echo "[$SHORT] done"

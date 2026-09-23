#!/usr/bin/env bash
set -uo pipefail
PGT=/home/padela/Scrivania/PGTNet
SRC=/home/padela/Scrivania/LLM_PPM/similar_prefix_temporal_fixed_sets_5050
PY=/home/padela/anaconda3/envs/graphgps/bin/python
[ -x "$PY" ] || PY=python3
mkdir -p "$PGT/new_logs" "$PGT/fts5050_splits"
$PY - <<'EOF' > /tmp/fts5050_order.txt
import json,os
logs=json.load(open('/home/padela/Scrivania/PGTNet/experiments_fts5050/logs.json'))
SRC='/home/padela/Scrivania/LLM_PPM/similar_prefix_temporal_fixed_sets_5050'
rows=[(os.path.getsize(f"{SRC}/{l['src']}_similar_prefix_temporal_fixed_sets.json"), l['short'], l['src']) for l in logs]
for sz,s,src in sorted(rows):
    print(s, src)
EOF
while read -r SHORT SRCBASE; do
  echo "--- $SHORT"
  $PY "$PGT/experiments_fts5050/build_fts5050.py" \
      --src "$SRC/${SRCBASE}_similar_prefix_temporal_fixed_sets.json" \
      --short "$SHORT" \
      --out_jsonl "$PGT/new_logs/${SHORT}_fts5050_preprocessed.jsonl" \
      --out_split "$PGT/fts5050_splits/${SHORT}.json" || echo "FAILED $SHORT"
done < /tmp/fts5050_order.txt
echo "[build_all] done $(date)"

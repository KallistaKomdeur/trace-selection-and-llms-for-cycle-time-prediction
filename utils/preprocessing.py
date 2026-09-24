import json
from pathlib import Path

import pandas as pd

from utils.log_schema import load_log_schema

def extract_timestamp_features(group, timestamp_col):
    """Adds 'timesincecasestart' (minutes) to every event in a single case."""
    group = group.sort_values(timestamp_col, ascending=True, kind="mergesort")
    start = group[timestamp_col].iloc[0]
    group["timesincecasestart"] = (group[timestamp_col] - start).dt.total_seconds() / 60
    return group

def build_event_features(group, timestamp_col, activity_col):
    """Builds the [activity, time_since_case_start] sequence for one case."""
    group = group.sort_values(timestamp_col)
    has_tsc = "timesincecasestart" in group.columns

    seq = []
    for row in group.itertuples(index=False):
        activity = getattr(row, activity_col)
        tsc = getattr(row, "timesincecasestart") if has_tsc else 0
        seq.append([activity, tsc])

    return seq

def safe_convert(obj):
    """Helper safe converter for different data types in logs"""
    return obj.item() if hasattr(obj, "item") else obj

def preprocess_log(log_name):
    """Reads logs/<log_name>/<log_name>.csv and writes the preprocessed JSONL next to it."""
    schema = load_log_schema(log_name)
    case_id_col = schema.case_id
    activity_col = schema.activity
    timestamp_col = schema.timestamp
    case_attr_cols = schema.case_attributes

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name
    input_file = log_dir / f"{log_name}.csv"
    output_file = log_dir / f"{log_name}_preprocessed.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(f"Log not found: {input_file}")

    data = pd.read_csv(input_file, encoding="latin-1", low_memory=False)
    available_cols = set(data.columns)
    has_ts = timestamp_col in available_cols
    has_act = activity_col in available_cols

    if has_ts:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce", utc=True)
        data = data.sort_values(timestamp_col).reset_index(drop=True)
        case_start = data.groupby(case_id_col)[timestamp_col].transform("min")
        data["timesincecasestart"] = (data[timestamp_col] - case_start).dt.total_seconds() / 60

    print("Building output")
    output = []

    if has_ts:
        case_end = data.groupby(case_id_col)[timestamp_col].max()
        case_order = case_end.sort_values().index
    else:
        case_order = list(data.groupby(case_id_col).groups.keys())

    grouped = data.groupby(case_id_col, sort=False)

    for cid in case_order:
        group = grouped.get_group(cid)
        total_time = ((group[timestamp_col].max() - group[timestamp_col].min()).total_seconds() / 60 if has_ts else 0)

        if has_ts:
            start_ts = group[timestamp_col].min().timestamp()
            end_ts = group[timestamp_col].max().timestamp()
        else:
            start_ts = None
            end_ts = None

        case_attrs = {c: group[c].iloc[0] for c in (case_attr_cols or [])}

        output.append({
            case_id_col: cid,
            **case_attrs,
            "ActTimeSeq": build_event_features(group, timestamp_col, activity_col) if has_act else [],
            "total_time": total_time,
            "start_ts": start_ts,
            "end_ts": end_ts
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for case in output:
            f.write(json.dumps(case, default=safe_convert) + "\n")

    return output_file
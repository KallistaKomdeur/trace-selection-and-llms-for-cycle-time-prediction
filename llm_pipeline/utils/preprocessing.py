import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from pathlib import Path
import json
from utils.log_schema import load_log_schema

def extract_timestamp_features(group, timestamp_col):
    """ Get general timestamp features like time since last event, time since case start, and how manieth event"""
    group = group.sort_values(timestamp_col, ascending=False, kind="mergesort")
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    group["timesincelastevent"] = tmp.fillna(pd.Timedelta(0)).dt.total_seconds() / 60
    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    group["timesincecasestart"] = tmp.dt.total_seconds() / 60
    group = group.sort_values(timestamp_col, ascending=True, kind="mergesort")
    group["event_nr"] = range(1, len(group) + 1)

    return group

def ent(data, col):
    """ Helper to get entropy"""
    p = data[col].value_counts(normalize=True)
    return stats.entropy(p)

def get_prev_resource(group, resource_col):
    """ Helper to get previous resource"""
    group["prev_resource"] = group[resource_col].shift(1).fillna("first")       # If first activity, there is no previous resource
    return group

def extract_resource_experience(group, case_id_col, activity_col, timestamp_col):
    """ 
    Get full resource experience, including (see inter-case features paper for definitions):
    - Entropy of activities of resource
    - Entropy of cases of resource
    - Entropy of handoffs
    - Busyness
    """
    group = group.reset_index(drop=True)
    n = len(group)
    ent_act = np.zeros(n)
    ent_case = np.zeros(n)
    ent_handoff = np.zeros(n)
    busyness = np.zeros(n)
    
    for i in range(n):
        hist = group.iloc[: i + 1]

        ent_act[i] = ent(hist, activity_col)
        ent_case[i] = ent(hist, case_id_col)
        ent_handoff[i] = ent(hist, "prev_resource")

        minutes = (hist[timestamp_col].max() - hist[timestamp_col].min()).total_seconds() / 60
        busyness[i] = len(hist) / minutes if minutes > 0 else 0.0
    
    group["ent_act"] = ent_act
    group["ent_case"] = ent_case
    group["ent_handoff"] = ent_handoff
    group["busyness"] = busyness

    return group

def build_event_features(group, timestamp_col, activity_col):
    """ Adds computed event features based on what could be computed."""
    group = group.sort_values(timestamp_col).reset_index(drop=True)
    cols = set(group.columns)

    activity_values = group[activity_col].values if activity_col in cols else None
    prev_resource_values = group["prev_resource"].values if "prev_resource" in cols else None
    
    if prev_resource_values is not None:
        prev_resource_values = prev_resource_values.astype(str)

    seq = []
    for idx in range(len(group)):
        row = group.iloc[idx]
        event_features = {}

        # If feature is selected in settings, add to event features
        if "timesincemidnight" in cols:
            event_features["timesincemidnight"] = row["timesincemidnight"]
        if "weekday" in cols:
            event_features["weekday"] = row["weekday"]
        if "month" in cols:
            event_features["month"] = row["month"]
        if "timesincelastevent" in cols:
            event_features["timesincelastevent"] = row["timesincelastevent"]
        if "timesincecasestart" in cols:
            event_features["timesincecasestart"] = row["timesincecasestart"]
        if "event_nr" in cols:
            event_features["event_nr"] = row["event_nr"]
        if "prev_resource" in cols:
            event_features["prev_resource"] = row["prev_resource"]
        if "ent_act" in cols:
            event_features["ent_act"] = row["ent_act"]
        if "ent_case" in cols:
            event_features["ent_case"] = row["ent_case"]
        if "ent_handoff" in cols:
            event_features["ent_handoff"] = row["ent_handoff"]
        if "busyness" in cols:
            event_features["busyness"] = row["busyness"]
        if "open_cases" in cols:
            event_features["open_cases"] = row["open_cases"]
        if "res_work_items" in cols:
            event_features["res_work_items"] = row["res_work_items"]
        if "res_cases" in cols:
            event_features["res_cases"] = row["res_cases"]
        if "res_unique_tasks" in cols:
            event_features["res_unique_tasks"] = row["res_unique_tasks"]
        if "res_unique_handoffs" in cols:
            event_features["res_unique_handoffs"] = row["res_unique_handoffs"]
        if "res_ratio_workitems_global" in cols:
            event_features["res_ratio_workitems_global"] = row["res_ratio_workitems_global"]
        if "res_ratio_workitems_resource" in cols:
            event_features["res_ratio_workitems_resource"] = row["res_ratio_workitems_resource"]
        if "res_ratio_task_specific" in cols:
            event_features["res_ratio_task_specific"] = row["res_ratio_task_specific"]
        if "res_ratio_handoff_specific" in cols:
            event_features["res_ratio_handoff_specific"] = row["res_ratio_handoff_specific"]
        if "res_work_items_per_min" in cols:
            event_features["res_work_items_per_min"] = row["res_work_items_per_min"]

        # Add activity frequencies
        if activity_values is not None:
            hist_activities = activity_values[:idx + 1]
            unique, counts = np.unique(hist_activities, return_counts=True)
            event_features["act_freq"] = dict(zip(unique, counts.tolist()))
        
        # Add handoff frequencies
        if prev_resource_values is not None:
            hist_handoffs = prev_resource_values[:idx + 1]
            unique, counts = np.unique(hist_handoffs, return_counts=True)
            event_features["handoff_freq"] = dict(zip(unique, counts.tolist()))

        # Add individual event and system state to activity sequence
        seq.append([row[activity_col], row["timesincecasestart"] if "timesincecasestart" in cols else 0, event_features])

    return seq

def safe_convert(obj):
    """ Helper safe converter for different data types in logs"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def preprocess_log(log_name, clean_version):
    """Main function to do all the inter-case and system state feature computations."""
    # Get column names from schema
    schema = load_log_schema(log_name)
    case_id_col = schema.case_id
    activity_col = schema.activity
    resource_col = schema.resource
    timestamp_col = schema.timestamp
    case_attr_cols = schema.case_attributes

    # Get correct file locations
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name
    
    if clean_version:
        input_file = log_dir / f"{log_name}_clean.csv"
        output_file = log_dir / f"{log_name}_clean_preprocessed.jsonl"
    else:
        input_file = log_dir / f"{log_name}.csv"
        output_file = log_dir / f"{log_name}_preprocessed.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(f"Log not found: {input_file}")

    # Read log data
    original_data = pd.read_csv(input_file, encoding="latin-1")
    original_data[timestamp_col] = pd.to_datetime(original_data[timestamp_col], errors='coerce', utc=True)

    # TODO bac has some weird stuff going on where instant activities only have a start date. Since I use end date, I copy this over
    if log_name == "bac":
        original_data["START_DATE"] = pd.to_datetime(original_data["START_DATE"], errors="coerce")
        original_data["END_DATE"] = pd.to_datetime(original_data["END_DATE"], errors="coerce")
        original_data["END_DATE"] = original_data["END_DATE"].fillna(original_data["START_DATE"])

    # Checks which features are available. Timestamp and activity are pretty much mandatory for anything useful
    available_cols = set(original_data.columns)
    has_ts = timestamp_col in available_cols
    has_res = resource_col in available_cols
    has_act = activity_col in available_cols

    data = original_data.copy()

    # Ensure chronological order of input events by timestamp
    if has_ts:
        print("Sorting input data by timestamp")
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True) # convert to datetime if not already
        data = data.sort_values(timestamp_col).reset_index(drop=True)

    if has_ts:
        print("Extracting timestamp features")
        data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
        data["weekday"] = data[timestamp_col].dt.weekday
        data["month"] = data[timestamp_col].dt.month

        ts_data = data.copy()
        ts_features = ts_data.groupby(case_id_col, group_keys=False).apply(lambda g: extract_timestamp_features(g, timestamp_col))
        
        for col in ["timesincelastevent", "timesincecasestart", "event_nr"]:
            data[col] = ts_features[col].values

    if has_res:
        print("Detecting handoffs")
        prev_res_data = data.copy()
        prev_res_features = prev_res_data.groupby(case_id_col, group_keys=False).apply(lambda g: get_prev_resource(g, resource_col))
        data["prev_resource"] = prev_res_features["prev_resource"].values

    if has_res and has_act and has_ts:
        print(f"Extracting resource experience features")
        data['_row_id'] = range(len(data))
    
        res_exp_data = data.copy().sort_values(timestamp_col)
        res_exp_features = res_exp_data.groupby(resource_col, group_keys=False).apply(lambda g: extract_resource_experience(g, case_id_col, activity_col, timestamp_col))
        data = data.merge(res_exp_features[["_row_id", "ent_act", "ent_case", "ent_handoff", "busyness"]], on="_row_id", how="left", suffixes=('', '_new'))

    if has_ts:
        print("Calculating workload features")
        case_windows = data.groupby(case_id_col)[timestamp_col].agg(start="min", end="max")
        data["open_cases"] = data[timestamp_col].apply(lambda t: ((case_windows["start"] <= t) & (case_windows["end"] > t)).sum())

    if has_res and has_act and has_ts:
        if '_row_id' not in data.columns:
            data['_row_id'] = range(len(data))
        
        print("Computing resource statistics")
        resource_stats = defaultdict(lambda: {
            "work_items": 0, "cases": set(), "tasks": defaultdict(int),
            "handoffs": defaultdict(int), "handoff_set": set(), "first_ts": None
        })

        global_cases_seen = set()
        sorted_data = data.sort_values(timestamp_col).copy()
        
        res_stats_cols = {"res_work_items": [], "res_cases": [], "res_unique_tasks": [],"res_unique_handoffs": [], "res_ratio_workitems_global": [],"res_ratio_workitems_resource": [], "res_ratio_task_specific": [],"res_ratio_handoff_specific": [], "res_work_items_per_min": []}
        
        # Compute system state values
        for i, row in sorted_data.iterrows():
            res = row[resource_col]
            cid = row[case_id_col]
            act = row[activity_col]
            ho = row["prev_resource"]
            ts = row[timestamp_col]

            global_cases_seen.add(cid)
            st = resource_stats[res]
            st["work_items"] += 1
            st["cases"].add(cid)
            st["tasks"][act] += 1
            st["handoffs"][ho] += 1
            st["handoff_set"].add(ho)
            st["first_ts"] = st["first_ts"] or ts

            n = st["work_items"]
            duration = (ts - st["first_ts"]).total_seconds() / 60

            res_stats_cols["res_work_items"].append(n)
            res_stats_cols["res_cases"].append(len(st["cases"]))
            res_stats_cols["res_unique_tasks"].append(len(st["tasks"]))
            res_stats_cols["res_unique_handoffs"].append(len(st["handoff_set"]))
            res_stats_cols["res_ratio_workitems_global"].append(n / len(global_cases_seen))
            res_stats_cols["res_ratio_workitems_resource"].append(n / len(st["cases"]))
            res_stats_cols["res_ratio_task_specific"].append(st["tasks"][act] / n)
            res_stats_cols["res_ratio_handoff_specific"].append(st["handoffs"][ho] / n)
            res_stats_cols["res_work_items_per_min"].append(n / duration if duration > 0 else 0)

        for col_name, col_values in res_stats_cols.items():
            sorted_data[col_name] = col_values
        
        data = data.merge(sorted_data[["_row_id"] + list(res_stats_cols.keys())], on="_row_id", how="left", suffixes=('', '_new'))

    if '_row_id' in data.columns:
        data = data.drop('_row_id', axis=1)

    print(f"Building output")
    output = []
    
    if has_ts:
        # order cases by their end (= latest) timestamp
        case_order = data.groupby(case_id_col)[timestamp_col].max().sort_values().index
    else:
        # fallback: preserve grouping order by case id
        case_order = list(data.groupby(case_id_col).groups.keys())

    for cid in case_order:
        group = data[data[case_id_col] == cid]
        total_time = ((group[timestamp_col].max() - group[timestamp_col].min()).total_seconds() / 60 if has_ts else 0)
        # absolute start/end timestamps for temporal splitting
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
            "ActTimeSeq": (build_event_features(group, timestamp_col, activity_col) if has_act else []),
            "total_time": total_time,
            "start_ts": start_ts,
            "end_ts": end_ts
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for case in output:
            f.write(json.dumps(case, default=safe_convert) + "\n")

    return output_file
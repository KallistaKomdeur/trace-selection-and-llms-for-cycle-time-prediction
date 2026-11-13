import pandas as pd
from utils.log_schema import load_log_schema

def clean_log(original_csv_path, clean_csv_path, log_name):
    """
    Cleans a log CSV by (taken from Weytjens & De Weert):
    - Normalizing timestamps
    - Removing duplicate events (= same case, activity, timestamp)
    - Removing the top 5% longest cases by duration
    - Filtering out cases that start before or end after a defined time window
    """

    # Load schema
    schema = load_log_schema(log_name)
    case_col = schema.case_id
    timestamp_col = schema.timestamp

    if not timestamp_col:
        raise ValueError("Timestamp column must be defined in schema")

    # Load and parse
    df = pd.read_csv(original_csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="raise")
    df[timestamp_col] = df[timestamp_col].dt.floor("s")

    # Remove duplicate events
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove top 5% longest cases
    durations = df.groupby(case_col)[timestamp_col].agg(["min", "max"])
    durations["duration_days"] = ((durations["max"] - durations["min"]).dt.total_seconds() / 86400)
    cutoff = durations["duration_days"].quantile(0.95)
    good_cases = durations[durations["duration_days"] <= cutoff].index
    df = df[df[case_col].isin(good_cases)].reset_index(drop=True)

    # Filter by time window 
    start_bound = pd.Timestamp("1999-01-01", tz="UTC")  # TODO change for selected dataset, now arbitrary start
    end_bound = pd.Timestamp("2012-03-01", tz="UTC")    # TODO change for selected dataset, now BPIC2012 end
    case_times = df.groupby(case_col)[timestamp_col].agg(["min", "max"])
    valid_cases = case_times[(case_times["min"] >= start_bound) & (case_times["max"] < end_bound)].index
    df = df[df[case_col].isin(valid_cases)].reset_index(drop=True)

    # Save
    df[timestamp_col] = df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(clean_csv_path, index=False)
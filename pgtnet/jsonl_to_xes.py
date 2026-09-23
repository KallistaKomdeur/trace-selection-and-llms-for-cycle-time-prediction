"""Convert preprocessed JSONL event logs in `new_logs/` to XES files PGTNet can consume.

Each JSONL line is one trace:
{
  "<case-id-key>": "...",          # case:concept:name | case | case_id | Case ID
  "ActTimeSeq": [["activity_name", time_since_case_start_min, {event_attr_dict}], ...],
  "total_time": minutes,
  "start_ts": epoch_seconds,
  "end_ts":   epoch_seconds,
  ...optional case-level keys (AMOUNT_REQ, LoanGoal, ...)
}

Per event we synthesise `time:timestamp = start_ts + ev[1] * 60` (minutes -> seconds).
We pull a small fixed set of numerical event attributes used by PGTNet's edge encoder
to keep edge feature dimensionality reasonable. The categorical event attribute is
`prev_resource` (present in every line). Case-level attributes are dataset-specific
and configured via DATASETS below.
"""

import json
import os
import argparse
from datetime import datetime, timezone

import pandas as pd
import pm4py


# Per-dataset config: case-id key, case attrs (categorical, numerical), output XES name.
DATASETS = {
    "bpi11_calib_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ["Age"],
        "xes_name": "Bpi11Calib.xes",
        "short_name": "bpi11_calib",
    },
    "bpi11_half_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ["Age"],
        "xes_name": "Bpi11Half.xes",
        "short_name": "bpi11_half",
    },
    "production_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": ["Part Desc."],
        "case_num_att": ["Work Order Qty"],
        "xes_name": "Production.xes",
        "short_name": "production",
    },
    "purchasing_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": [],
        "case_num_att": [],
        "xes_name": "Purchasing.xes",
        "short_name": "purchasing",
    },
    "bpi12w_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": [],
        "case_num_att": ["AMOUNT_REQ"],
        "xes_name": "Bpi12w.xes",
        "short_name": "bpi12w",
    },
    "confidential_preprocessed.jsonl": {
        "case_id_key": "caseid",
        "case_attributes": [],
        "case_num_att": [],
        "xes_name": "Confidential.xes",
        "short_name": "confidential",
    },
    "helpdesk_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": [],
        "case_num_att": [],
        "xes_name": "HelpDeskV2.xes",
        "short_name": "helpdesk",
    },
    "bpic2011_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ["Age"],
        "xes_name": "BPIC2011.xes",
        "short_name": "bpic2011",
    },
    "bpic2015_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": [],
        "xes_name": "BPIC2015.xes",
        "short_name": "bpic2015",
    },
    "bpic2012_preprocessed.jsonl": {
        "case_id_key": "case",
        "case_attributes": [],
        "case_num_att": ["AMOUNT_REQ"],
        "xes_name": "BPIC12V2.xes",
        "short_name": "bpic2012",
    },
    "bpic2017_preprocessed.jsonl": {
        "case_id_key": "case",
        "case_attributes": ["LoanGoal"],
        "case_num_att": ["RequestedAmount"],
        "xes_name": "BPIC17.xes",
        "short_name": "bpic2017",
    },
    "bpic2020_domestic_declarations_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ["Amount"],
        "xes_name": "BPIC20DomesticV2.xes",
        "short_name": "bpic2020_domestic",
    },
    "bpic2020_international_declarations_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ["Amount"],
        "xes_name": "BPIC20InternationalV2.xes",
        "short_name": "bpic2020_international",
    },
    "bpic2020_prepaid_travel_costs_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ["RequestedAmount"],
        "xes_name": "BPIC20Prepaid.xes",
        "short_name": "bpic2020_prepaid",
    },
    "bpic2020_request_for_payment_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ["RequestedAmount"],
        "xes_name": "BPIC20RFP.xes",
        "short_name": "bpic2020_rfp",
    },
    "bpic2020_travel_permit_data_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": ["Overspent"],
        "case_num_att": [],
        "xes_name": "BPIC20Permit.xes",
        "short_name": "bpic2020_permit",
    },
    "hospital_billing_preprocessed.jsonl": {
        "case_id_key": "case_id",
        "case_attributes": ["gender"],
        "case_num_att": ["age"],
        "xes_name": "HospitalBilling.xes",
        "short_name": "hospital_billing",
    },
    "traffic_fines_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": ["vehicleClass"],
        "case_num_att": ["article"],
        "xes_name": "TrafficFinesV2.xes",
        "short_name": "traffic_fines",
    },
    "bpic2011_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ['Age'],
        "xes_name": "BPIC2011.xes",
        "short_name": "bpic2011_fts5050",
    },
    "bpic2012_fts5050_preprocessed.jsonl": {
        "case_id_key": "case",
        "case_attributes": [],
        "case_num_att": ['AMOUNT_REQ'],
        "xes_name": "BPIC12V2.xes",
        "short_name": "bpic2012_fts5050",
    },
    "bpic2015_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": [],
        "xes_name": "BPIC2015.xes",
        "short_name": "bpic2015_fts5050",
    },
    "bpic2020_domestic_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ['Amount'],
        "xes_name": "BPIC20DomesticV2.xes",
        "short_name": "bpic2020_domestic_fts5050",
    },
    "bpic2020_international_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ['Amount'],
        "xes_name": "BPIC20InternationalV2.xes",
        "short_name": "bpic2020_international_fts5050",
    },
    "bpic2020_prepaid_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ['RequestedAmount'],
        "xes_name": "BPIC20Prepaid.xes",
        "short_name": "bpic2020_prepaid_fts5050",
    },
    "bpic2020_rfp_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ['RequestedAmount'],
        "xes_name": "BPIC20RFP.xes",
        "short_name": "bpic2020_rfp_fts5050",
    },
    "bpic2020_permit_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": ['Overspent'],
        "case_num_att": [],
        "xes_name": "BPIC20Permit.xes",
        "short_name": "bpic2020_permit_fts5050",
    },
    "gen_baseline_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": ['Department', 'TriageLevel'],
        "case_num_att": ['Age'],
        "xes_name": "baseline.xes",
        "short_name": "gen_baseline_fts5050",
    },
    "gen_drifted_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": ['Department', 'TriageLevel'],
        "case_num_att": ['Age'],
        "xes_name": "baseline_drifted.xes",
        "short_name": "gen_drifted_fts5050",
    },
    "helpdesk_fts5050_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": [],
        "case_num_att": [],
        "xes_name": "HelpDeskV2.xes",
        "short_name": "helpdesk_fts5050",
    },
    "hospital_billing_fts5050_preprocessed.jsonl": {
        "case_id_key": "case_id",
        "case_attributes": ['gender'],
        "case_num_att": ['age'],
        "xes_name": "HospitalBilling.xes",
        "short_name": "hospital_billing_fts5050",
    },
    "bpic2011_full_fts5050_preprocessed.jsonl": {
        "case_id_key": "Case ID",
        "case_attributes": [],
        "case_num_att": ['Age'],
        "xes_name": "BPIC2011.xes",
        "short_name": "bpic2011_full_fts5050",
    },
    "bpi12w_full_fts5050_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": [],
        "case_num_att": ['AMOUNT_REQ'],
        "xes_name": "Bpi12w.xes",
        "short_name": "bpi12w_full_fts5050",
    },
    "bpi12w_fts5050_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": [],
        "case_num_att": ['AMOUNT_REQ'],
        "xes_name": "Bpi12w.xes",
        "short_name": "bpi12w_fts5050",
    },
    "bpic2012_full_fts5050_preprocessed.jsonl": {
        "case_id_key": "case",
        "case_attributes": [],
        "case_num_att": ['AMOUNT_REQ'],
        "xes_name": "BPIC12V2.xes",
        "short_name": "bpic2012_full_fts5050",
    },
    "hospital_billing_full_fts5050_preprocessed.jsonl": {
        "case_id_key": "case_id",
        "case_attributes": ['gender'],
        "case_num_att": ['age'],
        "xes_name": "HospitalBilling.xes",
        "short_name": "hospital_billing_full_fts5050",
    },
    "helpdesk_full_fts5050_preprocessed.jsonl": {
        "case_id_key": "case:concept:name",
        "case_attributes": [],
        "case_num_att": [],
        "xes_name": "HelpDeskV2.xes",
        "short_name": "helpdesk_full_fts5050",
    },
}

# Numerical event attributes pulled from each event's dict (small, informative subset).
# Keeping it small controls edge feature dim. Skipping `act_freq`/`handoff_freq` (dicts).
EVENT_NUM_ATT = [
    "open_cases",
    "busyness",
    "ent_act",
    "ent_case",
    "res_work_items",
    "res_unique_tasks",
]
# Single categorical event attribute available in every JSONL.
EVENT_CAT_ATT = ["prev_resource"]


def jsonl_to_dataframe(jsonl_path, cfg):
    """Read a JSONL file, build a flat event-level DataFrame compatible with pm4py write_xes."""
    rows = []
    cid_key = cfg["case_id_key"]
    case_cat = cfg["case_attributes"]
    case_num = cfg["case_num_att"]

    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            cid = str(rec[cid_key])
            start_ts = float(rec["start_ts"])
            case_extra = {f"case:{k}": rec.get(k) for k in case_cat + case_num}
            for ev in rec["ActTimeSeq"]:
                act, t_min, attr_dict = ev
                # synth absolute timestamp; ev[1] is minutes since case start
                ts = datetime.fromtimestamp(start_ts + float(t_min) * 60.0, tz=timezone.utc)
                row = {
                    "case:concept:name": cid,
                    "concept:name": str(act),
                    "lifecycle:transition": "complete",
                    "time:timestamp": ts,
                }
                row.update(case_extra)
                # categorical event attrs
                for k in EVENT_CAT_ATT:
                    row[k] = str(attr_dict.get(k, ""))
                # numerical event attrs
                for k in EVENT_NUM_ATT:
                    v = attr_dict.get(k)
                    row[k] = float(v) if v is not None else None
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


def write_xes(df, out_path):
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    df = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    pm4py.write_xes(df, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="Input JSONL filename in new_logs/")
    p.add_argument(
        "--out_dir",
        default="/home/padela/Scrivania/PGTNet/raw_dataset",
        help="Where to write the .xes file (defaults to raw_dataset/)",
    )
    args = p.parse_args()

    if args.jsonl not in DATASETS:
        raise SystemExit(f"Unknown dataset: {args.jsonl}. Add it to DATASETS dict.")
    cfg = DATASETS[args.jsonl]
    in_path = os.path.join("/home/padela/Scrivania/PGTNet/new_logs", args.jsonl)
    out_path = os.path.join(args.out_dir, cfg["xes_name"])

    print(f"[reading]  {in_path}")
    df = jsonl_to_dataframe(in_path, cfg)
    print(f"[stats]    {df['case:concept:name'].nunique()} cases, {len(df)} events")
    print(f"[writing]  {out_path}")
    os.makedirs(args.out_dir, exist_ok=True)
    write_xes(df, out_path)
    print("[done]")


if __name__ == "__main__":
    main()

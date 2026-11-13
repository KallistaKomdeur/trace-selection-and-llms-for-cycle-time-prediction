"""Shared loader of the remaining-time campaign results (one row per model, strategy,
repetition, log). Used by build_recap_xlsx.py and build_results_xlsx.py."""
import re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
PGT = Path("/path/to/PGTNet/experiments_fts5050")
FULL = {"bpic2012": 2618, "bpic2011": 208, "bpic2020_domestic_declarations": 1571,
        "bpic2020_international_declarations": 1290, "bpic2020_prepaid_travel_costs": 419,
        "bpic2020_request_for_payment": 1362, "bpic2020_travel_permit_data": 1413,
        "gen_baseline": 500, "gen_drifted": 500}
LOGS = list(FULL)
SHORT_LOG = {"bpic2012": "BPI12", "bpic2011": "BPI11", "bpic2020_domestic_declarations": "BPI20_dd",
             "bpic2020_international_declarations": "BPI20_id", "bpic2020_prepaid_travel_costs": "BPI20_ptc",
             "bpic2020_request_for_payment": "BPI20_rfp", "bpic2020_travel_permit_data": "BPI20_tpd",
             "gen_baseline": "Gen_base", "gen_drifted": "Gen_drift"}
STRAT = {"random": "Random", "lambda0": "λ=0", "similar_prefix_temporal": "λ=0.5", "similar_prefix": "λ=1"}
MODEL = {("anthropic", "claude-sonnet-5-5"): "Claude Sonnet 5.5", ("zai", "glm-5.3-flash"): "GLM-5.3-Flash",
         ("llamacpp", "primary"): "Qwen3.5-9B"}
PGT_SHORT = {"bpic2011": "bpic2011", "bpic2012": "bpic2012", "bpic2020_domestic": "bpic2020_domestic_declarations",
             "bpic2020_international": "bpic2020_international_declarations",
             "bpic2020_prepaid": "bpic2020_prepaid_travel_costs", "bpic2020_rfp": "bpic2020_request_for_payment",
             "bpic2020_permit": "bpic2020_travel_permit_data", "gen_baseline": "gen_baseline", "gen_drifted": "gen_drifted"}
PGT_SETS = {"fewshot_remaining": "λ=0.5", "fewshot_l0_remaining": "λ=0", "fewshot_cf_remaining": "λ=1"}
NOTES = {"Claude Sonnet 5.5": "max_tokens 256: ~30% of the cases skipped (reasoning in the text), reduced n",
         "Qwen3.5-9B": "16k context: bpic2011 almost entirely skipped (context overflow)"}
PAT = re.compile(r"results_(?P<prov>[a-z]+)_(?P<model>[^_]+(?:-[^_]+)*)_(?P<mode>random|lambda0|similar_prefix_temporal|similar_prefix)_single_ref_remaining(?:_rep(?P<rep>\d+))?\.csv$")


def load_rows():
    rows = []
    for f in sorted(BASE.glob("results_*_single_ref_remaining*.csv")):
        m = PAT.match(f.name)
        if not m:
            continue
        label = MODEL.get((m["prov"], m["model"]), f"{m['prov']}/{m['model']}")
        rep = int(m["rep"] or 0)
        for _, r in pd.read_csv(f).iterrows():
            rows.append(dict(modello=label, strategia=STRAT[m["mode"]], rep=rep, log=r.log, n=int(r.n),
                             n_pieno=FULL[r.log], r2=float(r.r2), mae_h=float(r.mae_h), ra=float(r.ra),
                             ra_lo=float(r.ra_lo), ra_hi=float(r.ra_hi), outlier=int(r.outliers),
                             nota=NOTES.get(label, "")))
    for d in sorted(PGT.glob("fewshot_*remaining*")):
        f = d / "results_fewshot.csv"
        mm = re.match(r"(fewshot_(?:l0_|cf_)?remaining)(?:_seed(\d+))?$", d.name)
        if not f.exists() or not mm:
            continue
        seed = int(mm.group(2) or 42)
        for _, r in pd.read_csv(f).iterrows():
            rows.append(dict(modello="PGTNet few-shot", strategia=PGT_SETS[mm.group(1)], rep=seed - 42,
                             log=PGT_SHORT[r.log], n=int(r.n), n_pieno=int(r.n_rows), r2=float(r.r2),
                             mae_h=float(r.mae_h), ra=float(r.ra), ra_lo=float(r.ra_lo), ra_hi=float(r.ra_hi),
                             outlier=int(r.outliers), nota=f"seed {seed}; BPI12 10% sample; untrainable sets skipped"))
    f = PGT / "results_fts5050_remaining.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            mm = re.match(r"(.+)_full_remaining(?:_seed(\d+))?$", r.log)
            short, seed = mm.group(1), int(mm.group(2) or 42)
            rows.append(dict(modello="PGTNet full-log", strategia="FULL LOG", rep=seed - 42, log=PGT_SHORT[short],
                             n=int(r.n), n_pieno=int(r.n_test), r2=float(r.r2), mae_h=float(r.mae_h), ra=float(r.ra),
                             ra_lo=float(r.ra_lo), ra_hi=float(r.ra_hi), outlier=int(r.outliers),
                             nota=f"seed {seed}; train = every non-test case of the log; target remaining time"))
    f = BASE / "pgtnet_full_paper.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            rows.append(dict(modello="PGTNet full-log (paper)", strategia="FULL LOG", rep=0, log=r.log, n=int(r.n),
                             n_pieno=FULL[r.log], r2=float(r.r2), mae_h=float("nan"), ra=float(r.ra),
                             ra_lo=float(r.ra) - float(r.ra_ci), ra_hi=float(r.ra) + float(r.ra_ci), outlier=0,
                             nota=f"copied from the previous paper, total-time protocol (not re-run under remaining time); source: {r.fonte}"))
    return pd.DataFrame(rows)

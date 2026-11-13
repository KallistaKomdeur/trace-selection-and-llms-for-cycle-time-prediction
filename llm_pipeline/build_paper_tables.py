"""Tables and number macros of the LLNCS paper (remaining-time protocol).

Reads the same rows as the xlsx builders (recap_data.load_rows) and writes into
LLM_project_llncs/tex/gen/:

  tab_llms.tex       3 LLMs x 4 selection strategies, R2 / RA per log and mean
  tab_pgtnet.tex     GLM (best and combined distance) vs PGTNet full-log and few-shot
  tab_stats.tex      Friedman test + Holm post-hoc (control = best configuration)
  tab_coverage.tex   valid test cases per log for the LLMs that skip cases
  numbers.tex        \\newcommand macros with the figures quoted in the prose

Only repetitions that cover all nine logs enter the means (a partially run
repetition would bias a per-log mean). Re-run after every new seed/repetition and
recompile the paper: prose numbers are macros, so they follow the data.

    python3 build_paper_tables.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from recap_data import load_rows  # noqa: E402

OUT = BASE / "LLM_project_llncs" / "tex" / "gen"
OUT.mkdir(parents=True, exist_ok=True)

ALL_LOGS = ["bpic2012", "bpic2011", "bpic2020_domestic_declarations",
            "bpic2020_international_declarations", "bpic2020_prepaid_travel_costs",
            "bpic2020_request_for_payment", "bpic2020_travel_permit_data",
            "gen_baseline", "gen_drifted"]
# The paper reports the seven real-life logs only (user decision 2026-11-12: the two
# synthetic logs are dropped). Completeness of a repetition is still judged on all nine.
LOGS = ALL_LOGS[:7]
N_LOGS_WORD = "seven"
SHORT = {"bpic2012": "BPI12", "bpic2011": "BPI11", "bpic2020_domestic_declarations": "BPI20\\_dd",
         "bpic2020_international_declarations": "BPI20\\_id",
         "bpic2020_prepaid_travel_costs": "BPI20\\_ptc", "bpic2020_request_for_payment": "BPI20\\_rfp",
         "bpic2020_travel_permit_data": "BPI20\\_tpd", "gen_baseline": "Synthetic",
         "gen_drifted": "Synth.\\ Drifted"}
LLMS = ["Claude Sonnet 5.5", "GLM-5.3-Flash", "Qwen3.5-9B"]
LLM_SHORT = {"Claude Sonnet 5.5": "Sonnet 5.5", "GLM-5.3-Flash": "GLM-5.3-Flash", "Qwen3.5-9B": "Qwen3.5-9B",
             "PGTNet full-log": "PGTNet", "PGTNet few-shot": "PGTNet few-shot"}
STRATS = ["Random", "λ=0", "λ=0.5", "λ=1"]
# compact labels for the two wide results tables
COL = {"bpic2012": "BPI12", "bpic2011": "BPI11", "bpic2020_domestic_declarations": "dd",
       "bpic2020_international_declarations": "id", "bpic2020_prepaid_travel_costs": "ptc",
       "bpic2020_request_for_payment": "rfp", "bpic2020_travel_permit_data": "tpd",
       "gen_baseline": "Syn.", "gen_drifted": "Syn.\\,D."}
ROW = {"Claude Sonnet 5.5": "Sonnet 5.5", "GLM-5.3-Flash": "GLM-5.3-F.", "Qwen3.5-9B": "Qwen3.5-9B",
       "PGTNet full-log": "PGTNet", "PGTNet few-shot": "PGTNet f.-s."}
SEL = {"Random": "Rnd", "λ=0": "$\\lambda$=0", "λ=0.5": "$\\lambda$=0.5", "λ=1": "$\\lambda$=1", "FULL LOG": "full"}
STRAT_TEX = {"Random": "Random", "λ=0": "$\\lambda=0$", "λ=0.5": "$\\lambda=0.5$", "λ=1": "$\\lambda=1$",
             "FULL LOG": "full log"}


# ----------------------------------------------------------------------------- data
def complete_reps(df):
    """Keep, per (model, strategy), only the repetitions that cover all nine logs,
    then restrict to the seven real-life logs reported in the paper."""
    cov = df.groupby(["modello", "strategia", "rep"]).log.nunique().reset_index()
    ok = cov[cov.log == len(ALL_LOGS)][["modello", "strategia", "rep"]]
    return df.merge(ok, on=["modello", "strategia", "rep"])[lambda d: d.log.isin(LOGS)]


def per_log(df):
    """Mean and std over repetitions, per (model, strategy, log)."""
    g = df.groupby(["modello", "strategia", "log"]).agg(
        r2=("r2", "mean"), r2_sd=("r2", "std"), ra=("ra", "mean"), ra_sd=("ra", "std"),
        n=("n", "min"), n_full=("n_pieno", "first"), reps=("rep", "nunique")).reset_index()
    return g


def config_means(pl):
    """Mean of the seven per-log values; std of the mean over repetitions, if any."""
    rows = []
    for (m, s), g in pl.groupby(["modello", "strategia"]):
        rows.append(dict(modello=m, strategia=s, r2=g.r2.mean(), ra=g.ra.mean(),
                         reps=int(g.reps.min()), n_logs=len(g)))
    return pd.DataFrame(rows)


def rep_means(df):
    """Mean over the seven logs of every complete repetition: the std of these is the
    'std over repetitions' quoted in the tables."""
    g = df.groupby(["modello", "strategia", "rep"]).agg(r2=("r2", "mean"), ra=("ra", "mean")).reset_index()
    return g.groupby(["modello", "strategia"]).agg(r2_sd=("r2", "std"), ra_sd=("ra", "std"),
                                                    reps=("rep", "nunique")).reset_index()


# ------------------------------------------------------------------------ statistics
def friedman_holm(pl, configs, metric):
    """Friedman test over the seven logs (blocks) and Holm post-hoc against the best
    configuration (Demsar 2006). configs: list of (model, strategy)."""
    k, N = len(configs), len(LOGS)
    mat = np.array([[pl[(pl.modello == m) & (pl.strategia == s) & (pl.log == lg)][metric].iloc[0]
                     for lg in LOGS] for (m, s) in configs])          # k x N
    ranks = np.array([stats.rankdata(-mat[:, j]) for j in range(N)]).T  # higher value = rank 1
    mean_rank = ranks.mean(axis=1)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(mean_rank ** 2) - k * (k + 1) ** 2 / 4)
    p_chi2 = stats.chi2.sf(chi2, k - 1)
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    best = int(np.argmin(mean_rank))
    se = math.sqrt(k * (k + 1) / (6 * N))
    comps = []
    for i in range(k):
        if i == best:
            continue
        z = (mean_rank[i] - mean_rank[best]) / se
        comps.append([i, z, 2 * stats.norm.sf(abs(z))])
    comps.sort(key=lambda c: c[2])
    m = len(comps)
    prev = 0.0
    for j, c in enumerate(comps):
        adj = min(1.0, max(prev, (m - j) * c[2]))
        c.append(adj)
        prev = adj
    holm = {c[0]: (c[1], c[2], c[3]) for c in comps}
    return dict(chi2=chi2, p_chi2=p_chi2, ff=ff, p_ff=p_ff, mean_rank=mean_rank, best=best, holm=holm,
                wins_vs_best=[(int((mat[i] > mat[best]).sum()), int((mat[i] == mat[best]).sum()))
                              for i in range(k)])


# ---------------------------------------------------------------------- formatting
def f2(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def pct(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}\\%"



def p_tex(p):
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def results_table(pl, rm, configs, label, caption, bold_scope=None, header_label="Predictor", with_std=True):
    """Two stacked panels (R2, then RA): one row per configuration, one column per log
    plus the mean over the seven logs. bold_scope: configs among which the best per
    column is bolded."""
    bold_scope = bold_scope or configs
    means = config_means(pl)
    lines = ["\\begin{table}[t!]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{2.5pt}", "\\scriptsize",
             "\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll|" + "r" * len(LOGS) + "|r}", "\\toprule"]
    def sd_mean(c, metric):
        sd = rm[(rm.modello == c[0]) & (rm.strategia == c[1])]
        return float(sd.iloc[0][metric + "_sd"]) if with_std and len(sd) and sd.iloc[0].reps > 1 else float("nan")

    def nz(x):
        return 0.0 if x is None or math.isnan(x) else float(x)

    def bold_set(vals, sds):
        """Best configuration plus every one whose mean +- std interval overlaps the best's."""
        b = max(vals, key=vals.get)
        lo_b, hi_b = vals[b] - sds[b], vals[b] + sds[b]
        return {c for c in vals if vals[c] + sds[c] >= lo_b and vals[c] - sds[c] <= hi_b}

    for metric, fmt, title in (("r2", f2, "$R^2$"), ("ra", pct, "$RA$")):
        best = {}
        for lg in LOGS:
            rows_ = {c: pl[(pl.modello == c[0]) & (pl.strategia == c[1]) & (pl.log == lg)].iloc[0] for c in bold_scope}
            vals = {c: float(r[metric]) for c, r in rows_.items()}
            sds = {c: nz(r[metric + "_sd"]) if with_std and r.reps > 1 else 0.0 for c, r in rows_.items()}
            best[lg] = bold_set(vals, sds)
        vals = {c: float(means[(means.modello == c[0]) & (means.strategia == c[1])][metric].iloc[0]) for c in bold_scope}
        sds = {c: nz(sd_mean(c, metric)) for c in bold_scope}
        best["mean"] = bold_set(vals, sds)
        lines.append(f"{header_label} & Sel. & " + " & ".join(COL[lg] for lg in LOGS) + f" & Mean {title} \\\\")
        lines.append("\\midrule")
        prev_model = None
        for (m, s) in configs:
            if prev_model is not None and m != prev_model:
                lines.append("\\addlinespace[2pt]")
            prev_model = m
            cells = []
            for lg in LOGS:
                r = pl[(pl.modello == m) & (pl.strategia == s) & (pl.log == lg)].iloc[0]
                txt = fmt(r[metric])
                csd = float(r[metric + "_sd"])
                if with_std and r.reps > 1 and not math.isnan(csd):
                    txt += f"{{\\tiny$\\pm${f2(csd) if metric == 'r2' else f'{100 * csd:.1f}'}}}"
                if (m, s) in best[lg]:
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
            mr = means[(means.modello == m) & (means.strategia == s)].iloc[0]
            sdv = sd_mean((m, s), metric)
            txt = fmt(mr[metric]) if math.isnan(sdv) else (
                f"{fmt(mr[metric])}{{\\tiny$\\pm${f2(sdv) if metric == 'r2' else f'{100 * sdv:.1f}'}}}")
            if (m, s) in best["mean"]:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
            lines.append(f"{ROW.get(m, m)} & {SEL[s]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if metric == "r2" else "\\bottomrule")
    lines += ["\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def stats_table(pl, configs, label, caption):
    res = {m: friedman_holm(pl, configs, m) for m in ("r2", "ra")}
    lines = ["\\begin{table}[t!]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{ll|ccc|ccc}", "\\toprule",
             " & & \\multicolumn{3}{c|}{$R^2$} & \\multicolumn{3}{c}{$RA$} \\\\",
             "Predictor & Sel. & rank & wins & $p_{\\mathrm{Holm}}$ & rank & wins & $p_{\\mathrm{Holm}}$ \\\\",
             "\\midrule"]
    order = sorted(range(len(configs)), key=lambda i: res["r2"]["mean_rank"][i] + res["ra"]["mean_rank"][i])
    for i in order:
        m, s = configs[i]
        cells = []
        for metric in ("r2", "ra"):
            r = res[metric]
            rank = f"{r['mean_rank'][i]:.2f}"
            if i == r["best"]:
                cells += [f"\\textbf{{{rank}}}", "--", "--"]
            else:
                w, t = r["wins_vs_best"][i]
                wins = f"{w + 0.5 * t:g}"
                z, p, adj = r["holm"][i]
                ptxt = p_tex(adj)
                if adj < 0.05:
                    ptxt = f"\\textbf{{{ptxt}}}"
                cells += [rank, wins, ptxt]
        lines.append(f"{LLM_SHORT.get(m, m)} & {STRAT_TEX[s]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    r2, ra = res["r2"], res["ra"]
    lines.append(f"\\multicolumn{{2}}{{l|}}{{Friedman $\\chi^2_F$ ($p$)}} & \\multicolumn{{3}}{{c|}}{{{r2['chi2']:.1f} ({p_tex(r2['p_chi2'])})}}"
                 f" & \\multicolumn{{3}}{{c}}{{{ra['chi2']:.1f} ({p_tex(ra['p_chi2'])})}} \\\\")
    lines.append(f"\\multicolumn{{2}}{{l|}}{{Iman--Davenport $F_F$ ($p$)}} & \\multicolumn{{3}}{{c|}}{{{r2['ff']:.1f} ({p_tex(r2['p_ff'])})}}"
                 f" & \\multicolumn{{3}}{{c}}{{{ra['ff']:.1f} ({p_tex(ra['p_ff'])})}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n", res


def coverage_table(pl, label, caption):
    configs = [(m, s) for m in ["Claude Sonnet 5.5", "Qwen3.5-9B"] for s in STRATS]
    first = pl[(pl.modello == "GLM-5.3-Flash")]
    tot = {lg: int(first[first.log == lg].n_full.iloc[0]) for lg in LOGS}
    lines = ["\\begin{table}[t!]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{3pt}", "\\footnotesize",
             "\\begin{tabular}{l|r|" + "r" * len(STRATS) + "|" + "r" * len(STRATS) + "}", "\\toprule",
             " & & \\multicolumn{4}{c|}{Sonnet 5.5} & \\multicolumn{4}{c}{Qwen3.5-9B} \\\\",
             "Event log & Cases & " + " & ".join(STRAT_TEX[s] for s in STRATS) + " & "
             + " & ".join(STRAT_TEX[s] for s in STRATS) + " \\\\", "\\midrule"]

    def num(n):
        return f"{n:,}".replace(",", "\\,")

    sums = {c: 0 for c in configs}
    for lg in LOGS:
        cells = []
        for c in configs:
            n = int(pl[(pl.modello == c[0]) & (pl.strategia == c[1]) & (pl.log == lg)].n.iloc[0])
            sums[c] += n
            cells.append(num(n))
        lines.append(f"{SHORT[lg]} & {num(tot[lg])} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    total = sum(tot.values())
    lines.append(f"All & {num(total)} & " + " & ".join(num(sums[c]) for c in configs) + " \\\\")
    lines.append("(share) & & " + " & ".join(f"{100 * sums[c] / total:.0f}\\%" for c in configs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


# --------------------------------------------------------------------------- main
def main():
    df = load_rows()
    df = df[df.modello != "PGTNet full-log (paper)"]
    df = complete_reps(df)
    pl = per_log(df)
    rm = rep_means(df)
    means = config_means(pl)

    def mean_of(m, s, metric):
        return float(means[(means.modello == m) & (means.strategia == s)][metric].iloc[0])

    def sd_of(m, s, metric):
        r = rm[(rm.modello == m) & (rm.strategia == s)]
        return float(r.iloc[0][metric + "_sd"]) if len(r) and r.iloc[0].reps > 1 else float("nan")

    def reps_of(m, s):
        r = rm[(rm.modello == m) & (rm.strategia == s)]
        return int(r.iloc[0].reps) if len(r) else 0

    # ---- Table: the three LLMs x four selection strategies
    llm_cfg = [(m, s) for m in LLMS for s in STRATS]
    pgt_full = ("PGTNet full-log", "FULL LOG")
    few_cfg = [("PGTNet few-shot", s) for s in ["λ=0", "λ=0.5", "λ=1"]]
    glm_reps = reps_of("GLM-5.3-Flash", "λ=0.5")
    cap_llms = ("$R^2$ and $RA$ of the three LLMs on each event log and on average over the seven logs, "
                "for the four selection strategies ($n=10$; Rnd = random). dd, id, ptc, rfp, tpd are the five BPI20 logs. "
                "GLM-5.3-Flash values are means over "
                f"{glm_reps} repetitions for $\\lambda\\in\\{{0,0.5,1\\}}$ ($\\pm$ std over the repetitions; the last column reports the std of the seven-log mean) and a single run for Rnd; "
                "Sonnet 5.5 and Qwen3.5-9B are single runs. Best value per column in bold, together with the values whose mean $\\pm$ std interval overlaps with it.")
    (OUT / "tab_llms.tex").write_text(results_table(pl, rm, llm_cfg, "tab:llms", cap_llms, header_label="LLM"))

    # ---- Table: GLM vs PGTNet
    full_reps = reps_of(*pgt_full)
    few_reps = min(reps_of(*c) for c in few_cfg)
    few_reps_max = max(reps_of(*c) for c in few_cfg)
    few_reps_txt = str(few_reps) if few_reps == few_reps_max else f"{few_reps} to {few_reps_max}"
    cmp_cfg = [("GLM-5.3-Flash", "λ=1"), ("GLM-5.3-Flash", "λ=0.5"), ("GLM-5.3-Flash", "λ=0"), pgt_full] + few_cfg
    cap_pgt = ("$R^2$ and $RA$ of GLM-5.3-Flash with control-flow ($\\lambda=1$), combined ($\\lambda=0.5$) and temporal ($\\lambda=0$) "
               "selection against PGTNet trained on the full training log and PGTNet few-shot, trained on the "
               f"same ten selected traces. Means over {glm_reps} repetitions (GLM), {full_reps} training seed"
               f"{'s' if full_reps > 1 else ''} (PGTNet full log) and {few_reps_txt} seeds (PGTNet few-shot), "
               "$\\pm$ std over repetitions (per log, and of the seven-log mean in the last column). Log abbreviations as in \\autoref{tab:llms}. Best value per column in bold, together with the values whose mean $\\pm$ std interval overlaps with it.")
    (OUT / "tab_pgtnet.tex").write_text(results_table(pl, rm, cmp_cfg, "tab:pgtnet", cap_pgt))

    # ---- Statistics
    cap_s1 = ("Friedman test over the seven event logs for the twelve LLM configurations of "
              "\\autoref{tab:llms}. \\emph{rank}: mean rank (1 = best); \\emph{wins}: logs on which the "
              "configuration beats the control (best mean rank, in bold; ties count 0.5); "
              "$p_{\\mathrm{Holm}}$: Holm-adjusted $p$ of the post-hoc comparison with the control "
              "(bold if $<0.05$).")
    t1, res1 = stats_table(pl, llm_cfg, "tab:stats_llms", cap_s1)
    cap_s2 = ("Friedman test over the seven event logs for the configurations of \\autoref{tab:pgtnet}. "
              "\\emph{rank}: mean rank (1 = best); \\emph{wins}: logs on which the configuration beats the "
              "control (best mean rank, in bold; ties count 0.5); $p_{\\mathrm{Holm}}$: Holm-adjusted $p$ of the "
              "post-hoc comparison with the control (bold if $<0.05$).")
    t2, res2 = stats_table(pl, cmp_cfg, "tab:stats_pgtnet", cap_s2)
    (OUT / "tab_stats.tex").write_text(t2)
    (OUT / "tab_stats_llms.tex").write_text(t1)   # not included in the paper, kept for reference

    # ---- Coverage
    cap_cov = ("Test cases with a valid prediction per event log. Sonnet 5.5 is skipped when the "
               "generation exceeds the 256-token output budget; Qwen3.5-9B when the prompt exceeds its "
               "16\\,384-token context. GLM-5.3-Flash answers on at least 97\\% of the cases of every log "
               "(99.5\\% overall).")
    (OUT / "tab_coverage.tex").write_text(coverage_table(pl, "tab:coverage", cap_cov))

    # ---- Macros for the prose
    M = ""
    for m in LLMS:
        key = {"Claude Sonnet 5.5": "son", "GLM-5.3-Flash": "glm", "Qwen3.5-9B": "qwen"}[m]
        for s, sk in zip(STRATS, ["Rnd", "Lzero", "Lhalf", "Lone"]):
            M += macro(f"{key}{sk}Rtwo", f2(mean_of(m, s, "r2")))
            M += macro(f"{key}{sk}RA", pct(mean_of(m, s, "ra")))
    M += macro("glmReps", str(glm_reps))
    M += macro("glmLoneRtwoSd", f2(sd_of("GLM-5.3-Flash", "λ=1", "r2")))
    M += macro("glmLoneRASd", f"{100 * sd_of('GLM-5.3-Flash', 'λ=1', 'ra'):.1f}")
    M += macro("pgtFullReps", str(full_reps))
    M += macro("pgtFullRtwo", f2(mean_of(*pgt_full, "r2")))
    M += macro("pgtFullRA", pct(mean_of(*pgt_full, "ra")))
    for s, sk in zip(["λ=0", "λ=0.5", "λ=1"], ["Lzero", "Lhalf", "Lone"]):
        M += macro(f"pgtFew{sk}Rtwo", f2(mean_of("PGTNet few-shot", s, "r2")))
        M += macro(f"pgtFew{sk}RA", pct(mean_of("PGTNet few-shot", s, "ra")))
    M += macro("pgtFewReps", str(few_reps))
    # selection gain: mean over the three LLMs of (lambda=1 - random)
    gains_r2 = [mean_of(m, "λ=1", "r2") - mean_of(m, "Random", "r2") for m in LLMS]
    gains_ra = [mean_of(m, "λ=1", "ra") - mean_of(m, "Random", "ra") for m in LLMS]
    M += macro("gainRtwo", f2(float(np.mean(gains_r2))))
    M += macro("gainRA", f"{100 * float(np.mean(gains_ra)):.0f}")
    # per-log wins of GLM lambda=1 over PGTNet full
    g = pl[(pl.modello == "GLM-5.3-Flash") & (pl.strategia == "λ=1")].set_index("log")
    p = pl[(pl.modello == pgt_full[0]) & (pl.strategia == pgt_full[1])].set_index("log")
    M += macro("glmWinsRtwo", str(int((g.loc[LOGS].r2 > p.loc[LOGS].r2).sum())))
    M += macro("glmWinsRA", str(int((g.loc[LOGS].ra > p.loc[LOGS].ra).sum())))
    M += macro("glmWinLogsRtwo", ", ".join(SHORT[lg] for lg in LOGS if g.loc[lg].r2 > p.loc[lg].r2) or "none")
    M += macro("glmWinLogsRA", ", ".join(SHORT[lg] for lg in LOGS if g.loc[lg].ra > p.loc[lg].ra) or "none")
    # per-log wins of GLM lambda=1 over the other two LLMs with lambda=1 (ties excluded)
    for other, key in (("Claude Sonnet 5.5", "Son"), ("Qwen3.5-9B", "Qwen")):
        o = pl[(pl.modello == other) & (pl.strategia == "λ=1")].set_index("log")
        for metric, mk in (("r2", "Rtwo"), ("ra", "RA")):
            M += macro(f"glmVs{key}Wins{mk}", str(int((g.loc[LOGS][metric].round(3) > o.loc[LOGS][metric].round(3)).sum())))
    # statistics
    for name, res, cfg in (("llm", res1, llm_cfg), ("cmp", res2, cmp_cfg)):
        for metric in ("r2", "ra"):
            r = res[metric]
            mk = "Rtwo" if metric == "r2" else "RA"
            b = cfg[r["best"]]
            M += macro(f"{name}Best{mk}", "PGTNet trained on the full log" if b[0] == "PGTNet full-log"
                       else f"{LLM_SHORT.get(b[0], b[0])} with {STRAT_TEX[b[1]]} selection")
            M += macro(f"{name}BestRank{mk}", f"{r['mean_rank'][r['best']]:.2f}")
            M += macro(f"{name}Chi{mk}", f"{r['chi2']:.1f}")
            # used inside math mode as $p\\cmpPRtwo$: emit "<0.001" or "=0.012"
            M += macro(f"{name}P{mk}", "<0.001" if r["p_chi2"] < 0.001 else f"={r['p_chi2']:.3f}")
            sig = [cfg[i] for i in range(len(cfg)) if i != r["best"] and r["holm"][i][2] < 0.05]
            M += macro(f"{name}NSig{mk}", str(len(sig)))
            M += macro(f"{name}NCmp{mk}", str(len(cfg) - 1))
    # Holm p of GLM l=1 vs PGTNet full and of l=0.5 vs best in the comparison block
    for metric in ("r2", "ra"):
        r = res2[metric]
        mk = "Rtwo" if metric == "r2" else "RA"
        for c, key in ((("GLM-5.3-Flash", "λ=1"), "glmLone"), (("GLM-5.3-Flash", "λ=0.5"), "glmLhalf"),
                       (("GLM-5.3-Flash", "λ=0"), "glmLzero"), (pgt_full, "pgtFull")):
            i = cmp_cfg.index(c)
            if i == r["best"]:
                M += macro(f"cmpHolm{key}{mk}", "control")
                M += macro(f"cmpRank{key}{mk}", f"{r['mean_rank'][i]:.2f}")
            else:
                M += macro(f"cmpHolm{key}{mk}", p_tex(r["holm"][i][2]))
                M += macro(f"cmpRank{key}{mk}", f"{r['mean_rank'][i]:.2f}")
    # coverage
    son = pl[pl.modello == "Claude Sonnet 5.5"]
    tot = int(son.groupby("strategia").n_full.sum().iloc[0])
    M += macro("sonCovMin", f"{100 * son.groupby('strategia').n.sum().min() / tot:.0f}")
    M += macro("sonCovMax", f"{100 * son.groupby('strategia').n.sum().max() / tot:.0f}")
    q11 = pl[(pl.modello == "Qwen3.5-9B") & (pl.log == "bpic2011")]
    M += macro("qwenBpiElevenMin", str(int(q11.n.min())))
    M += macro("qwenBpiElevenMax", str(int(q11.n.max())))
    M += macro("cmpNCfg", {6: "six", 7: "seven", 8: "eight"}.get(len(cmp_cfg), str(len(cmp_cfg))))
    M += macro("tablesDate", pd.Timestamp.today().strftime("%Y-%m-%d"))
    (OUT / "numbers.tex").write_text(M)

    print(f"-> {OUT}")
    print(means.round(3).sort_values(['modello', 'strategia']).to_string(index=False))
    for name, res in (("LLM block", res1), ("comparison block", res2)):
        for metric in ("r2", "ra"):
            r = res[metric]
            print(f"{name} {metric}: chi2={r['chi2']:.1f} p={r['p_chi2']:.2g} best={r['best']} "
                  f"ranks={[round(x, 2) for x in r['mean_rank']]} holm={[round(v[2], 3) for k, v in sorted(r['holm'].items())]}")


if __name__ == "__main__":
    main()

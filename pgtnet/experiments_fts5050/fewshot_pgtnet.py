#!/usr/bin/env python
"""Few-shot PGTNet: one training run per test case, on that case's own examples.

The fixed sets give each test trace the 10 example traces the LLM receives in its
prompt. This trains PGTNet from scratch on exactly those 10 traces and predicts
that one test prefix, so the two predictors see the same evidence.

Nothing is re-selected: train = the entry's "examples", test = the entry's
"test_case" at its own prefix_length, target = its "true_total_time".

Graph construction reuses GTconvertor's own functions, so the features are the
ones the per-log PGTNet runs use; only the driver differs (one process looping
over sets instead of a shell pipeline per set, which would cost ~2 min of
overhead per run).
"""
import argparse, json, os, pickle, shutil, sys, time, warnings
from datetime import datetime, timezone

PGT = "/home/padela/Scrivania/PGTNet"
GPS = f"{PGT}/GraphGPS"
for p in (GPS, PGT, os.path.dirname(PGT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
from pm4py.objects.log.obj import EventLog, Trace, Event

import GTconvertor as GT

warnings.filterwarnings("ignore")

EVENT_NUM_ATT = ["open_cases", "busyness", "ent_act", "ent_case",
                 "res_work_items", "res_unique_tasks"]
EVENT_CAT_ATT = ["prev_resource"]
SRC_DIR = "/home/padela/Scrivania/LLM_PPM/similar_prefix_temporal_fixed_sets_5050"


# ----------------------------------------------------------------- log building
def _events(rec, cid, case_extra):
    """One pm4py Trace plus the flat rows the global-stat functions need."""
    start = float(rec["start_ts"])
    tr = Trace(attributes={"concept:name": cid, **case_extra})
    rows = []
    for act, t_min, *rest in rec["ActTimeSeq"]:
        attrs = rest[0] if rest else {}
        ts = datetime.fromtimestamp(start + float(t_min) * 60.0, tz=timezone.utc)
        ev = {"concept:name": str(act), "lifecycle:transition": "complete",
              "time:timestamp": ts}
        for k in EVENT_CAT_ATT:
            ev[k] = str(attrs.get(k, ""))
        for k in EVENT_NUM_ATT:
            v = attrs.get(k)
            ev[k] = float(v) if v is not None else np.nan
        tr.append(Event(ev))
        row = {"case:concept:name": cid, **ev}
        row.update({f"case:{k}": v for k, v in case_extra.items()})
        rows.append(row)
    return tr, rows


def build_entry(entry, cid_key, case_cat, case_num):
    """Return (train_log, test_log, full_log, train_df, full_df, case_spec, cid, pl)."""
    exs, tc = entry["examples"], entry["test_case"]
    pl = entry["prefix_length"]
    train_log, test_log, full_log = EventLog(), EventLog(), EventLog()
    train_rows, full_rows, spec = [], [], {}

    for ex in exs:
        cid = str(ex[cid_key])
        extra = {k: ex.get(k) for k in case_cat + case_num}
        tr, rows = _events(ex, cid, extra)
        train_log.append(tr)
        full_log.append(tr)
        train_rows += rows
        full_rows += rows
        spec[cid] = {"target_days": float(ex["total_time"]) / 1440.0,
                     "prefix_lengths": None}

    tcid = str(tc[cid_key])
    extra = {k: tc.get(k) for k in case_cat + case_num}
    tr, rows = _events(tc, tcid, extra)
    test_log.append(tr)
    full_log.append(tr)     # after the examples: get_global_stat_func indexes log[:n_train]
    full_rows += rows
    spec[tcid] = {"target_days": float(tc["true_total_time"]) / 1440.0,
                  "prefix_lengths": [pl]}

    return (train_log, test_log, full_log,
            pd.DataFrame(train_rows), pd.DataFrame(full_rows), spec, tcid, pl)


def convert(entry, cid_key, case_cat, case_num):
    """Build the train and test graphs for one fixed set. None if unusable."""
    (train_log, test_log, full_log, train_df, full_df,
     spec, tcid, pl) = build_entry(entry, cid_key, case_cat, case_num)

    case_cat_full, case_num_full = GT.case_full_func(case_cat, case_num)
    # node_class_dict is built from the full log, which is PGTNet's own behaviour and
    # is what lets a test prefix contain an activity absent from the 10 examples.
    (node_class_dict, max_case_df, max_active_cases, min_num_list, max_num_list,
     ev_min, ev_max, att_enc, case_enc, node_dim, edge_dim,
     avg_num_list, ev_avg) = GT.get_global_stat_func(
        train_log, train_df, case_num_full, EVENT_NUM_ATT, EVENT_CAT_ATT,
        case_cat_full, full_df, full_log)

    # get_global_stat_func only registers a class when it appears as the *source* of a
    # directly-follows pair, so an activity that only ever occurs last is missing. With
    # thousands of traces that never bites; with 11 it does, and the lookup raises.
    for trace in full_log:
        for ev in trace:
            cls = (ev.get("concept:name"), ev.get("lifecycle:transition"))
            if cls not in node_class_dict:
                node_class_dict[cls] = len(node_class_dict)

    starts = sorted(t[0].get("time:timestamp") for t in full_log)
    ends = sorted(t[len(t) - 1].get("time:timestamp") for t in full_log)
    max_time_norm = max(spec[str(e[cid_key])]["target_days"] for e in entry["examples"])
    if max_time_norm <= 0:
        return None

    common = dict(case_attributes=case_cat, case_encoder_list=case_enc,
                  case_num_att=case_num, min_num_list=min_num_list,
                  max_num_list=max_num_list, event_attributes=EVENT_CAT_ATT,
                  event_num_att=EVENT_NUM_ATT, target_normalization=True,
                  max_time_norm=max_time_norm, target_variable="total_time",
                  node_class_dict=node_class_dict, edge_dim=edge_dim,
                  max_case_df=max_case_df, sorted_start_dates=starts,
                  sorted_end_dates=ends, max_active_cases=max_active_cases,
                  attribute_encoder_list=att_enc, event_min_num_list=ev_min,
                  event_max_num_list=ev_max, avg_num_list=avg_num_list,
                  event_avg_num_list=ev_avg, case_spec=spec)

    _, idx, train_graphs = GT.graph_conversion_func(train_log, [], 0, [], **common)
    _, _, test_graphs = GT.graph_conversion_func(test_log, [], 0, [], **common)
    if not train_graphs or not test_graphs:
        return None
    # GTconvertor stores y as a 0-dim tensor. With hundreds of graphs per split that
    # collates fine, but a split holding a single graph comes back 0-dim while the
    # others come back shape [1], and PyG then refuses to concatenate the mix.
    # ...and `pl`, a plain int, comes back as a tensor from a many-graph split but as an
    # int from a one-graph split, which collate then cannot mix either.
    for g in train_graphs + test_graphs:
        g.y = g.y.reshape(1)
        g.pl = torch.tensor([int(g.pl)])
    return dict(train=train_graphs, test=test_graphs, node_dim=len(node_class_dict),
                edge_dim=edge_dim, norm=max_time_norm, cid=tcid, pl=pl,
                y_true_days=spec[tcid]["target_days"])


# ----------------------------------------------------------------- graphgym glue
def graph_dims(graphs):
    """Encoder table sizes. Undersizing these gives an async CUDA device assert."""
    n_nodes = max(int(g.x.shape[0]) for g in graphs)
    deg_in = deg_out = 0
    for g in graphs:
        if g.edge_index.numel():
            deg_out = max(deg_out, int(torch.bincount(g.edge_index[0]).max()))
            deg_in = max(deg_in, int(torch.bincount(g.edge_index[1]).max()))
    return n_nodes, deg_in, deg_out


class Runner:
    def __init__(self, cfg_path, dataset_format, scratch, max_epoch, patience,
                 min_rel_delta, seed=42):
        from torch_geometric.graphgym.config import cfg, set_cfg
        import graphgps  # noqa: F401  registers the custom modules
        self.cfg = cfg
        set_cfg(cfg)
        cfg.merge_from_file(cfg_path)
        cfg.dataset.format = dataset_format
        cfg.dataset.dir = scratch
        cfg.out_dir = os.path.join(scratch, "out")
        cfg.run_dir = os.path.join(scratch, "out", "0")
        # graphgym reads cfg.accelerator in create_model and cfg.device elsewhere;
        # both default to "auto", which torch.device() rejects.
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.device = dev
        cfg.accelerator = dev
        cfg.num_workers = 0
        cfg.num_threads = 4
        cfg.seed = seed
        cfg.train.enable_ckpt = False
        cfg.train.ckpt_best = False
        cfg.optim.max_epoch = max_epoch
        cfg.optim.num_warmup_epochs = min(10, max_epoch // 10)
        os.makedirs(cfg.run_dir, exist_ok=True)
        self.ds_dir = os.path.join(scratch, dataset_format.split("-", 1)[1])
        self.raw = os.path.join(self.ds_dir, "raw")
        self.processed = os.path.join(self.ds_dir, "processed")
        self.max_epoch, self.patience, self.min_rel_delta = max_epoch, patience, min_rel_delta
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _write(self, train_graphs, test_graphs):
        shutil.rmtree(self.processed, ignore_errors=True)
        os.makedirs(self.raw, exist_ok=True)
        os.makedirs(self.processed, exist_ok=True)
        # No validation split exists with 10 examples: we train for a fixed budget and
        # keep the last weights, so this slot is a placeholder the loop never reads.
        for name, graphs in (("train", train_graphs), ("val", train_graphs[:1]),
                             ("test", test_graphs)):
            with open(os.path.join(self.raw, f"{name}.pickle"), "wb") as f:
                pickle.dump(graphs, f)

    def run(self, conv):
        from torch_geometric.graphgym.loader import create_loader
        from torch_geometric.graphgym.model_builder import create_model
        from torch_geometric import seed_everything

        cfg = self.cfg
        self._write(conv["train"], conv["test"])
        n_nodes, deg_in, deg_out = graph_dims(conv["train"] + conv["test"])
        cfg.dataset.node_encoder_num_types = conv["node_dim"] + 1
        cfg.posenc_GraphormerBias.num_spatial_types = n_nodes + 2
        cfg.posenc_GraphormerBias.num_in_degrees = deg_in + 2
        cfg.posenc_GraphormerBias.num_out_degrees = deg_out + 2
        os.environ["PGTNET_EDGE_IN_DIM"] = str(conv["edge_dim"])

        seed_everything(cfg.seed)
        loaders = create_loader()
        model = create_model().to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.base_lr,
                                weight_decay=cfg.optim.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epoch)
        lossf = torch.nn.L1Loss()

        best, bad, epochs = float("inf"), 0, 0
        for epoch in range(self.max_epoch):
            model.train()
            tot, n = 0.0, 0
            for batch in loaders[0]:
                batch = batch.to(self.device)
                opt.zero_grad()
                pred, true = model(batch)
                loss = lossf(pred.reshape(-1), true.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * batch.num_graphs
                n += batch.num_graphs
            sched.step()
            epochs = epoch + 1
            cur = tot / max(n, 1)
            if cur < best * (1 - self.min_rel_delta):
                best, bad = cur, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loaders[2]:
                batch = batch.to(self.device)
                pred, true = model(batch)
                # squeeze(-1) would collapse a single-graph batch to a scalar
                for p, t, c, l in zip(pred.reshape(-1).tolist(),
                                      true.reshape(-1).tolist(),
                                      batch.cid, batch.pl.reshape(-1).tolist()):
                    preds.append((str(c), int(l), float(t), float(p)))
        out = dict(train_loss=best, epochs=epochs, preds=preds)
        del model, opt, loaders
        torch.cuda.empty_cache()
        return out


# ----------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="short name from logs.json")
    ap.add_argument("--max_epoch", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_rel_delta", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="stop after N sets (0 = all)")
    ap.add_argument("--sample_frac", type=float, default=1.0,
                    help="score a uniform random fraction of the test sets (seed 42), for "
                         "logs where one training per test case is too slow to run in full")
    ap.add_argument("--sets", choices=("temporal", "controlflow"), default="temporal",
                    help="which retrieval to train on: the 50/50 sets, or the original "
                         "control-flow-only ones in LLM_PPM/logs/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = {l["short"]: l for l in json.load(open(f"{PGT}/experiments_fts5050/logs.json"))}[a.log]
    if a.sets == "temporal":
        src = f"{SRC_DIR}/{meta['src']}_similar_prefix_temporal_fixed_sets.json"
        sub = "fewshot"
    else:
        # Both retrievals sample the same test cases at the same prefix lengths, so the
        # two campaigns differ only in the ten traces each run is trained on.
        src = (f"/home/padela/Scrivania/LLM_PPM/logs/{meta['src']}/"
               f"{meta['src']}_similar_prefix_fixed_sets.json")
        sub = "fewshot_cf"
    if not os.path.exists(src):
        raise SystemExit(f"fixed sets non trovati: {src}")
    out_dir = a.out or f"{PGT}/experiments_fts5050/{sub}/{a.log}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") in ("ok", "unusable"):
                        done.add(r["set_index"])
                except Exception:
                    pass
    if done:
        print(f"[{a.log}] resuming: {len(done)} sets already predicted", flush=True)

    scratch = f"/tmp/claude-1000/fewshot_{a.log}"
    shutil.rmtree(scratch, ignore_errors=True)
    runner = Runner(f"{PGT}/training_configs/{meta['cfg']}.yaml",
                    f"PyG-EVENT{meta['dataset']}", scratch,
                    a.max_epoch, a.patience, a.min_rel_delta)

    sys.path.insert(0, f"{PGT}/experiments_fts5050")
    from build_fts5050 import iter_entries, CID_KEYS

    keep = None
    if a.sample_frac < 1.0:
        n_all = sum(1 for _ in iter_entries(src))
        rng = np.random.default_rng(42)
        keep = set(rng.choice(n_all, size=max(1, round(n_all * a.sample_frac)),
                              replace=False).tolist())
        print(f"[{a.log}] campiono {len(keep)}/{n_all} set ({a.sample_frac:.0%}, seed 42)",
              flush=True)

    t0 = time.time()
    n_ok = n_skip = 0
    with open(out_path, "a") as fout:
        for i, entry in enumerate(iter_entries(src)):
            if i in done or (keep is not None and i not in keep):
                continue
            if a.limit and n_ok + n_skip >= a.limit:
                break
            cid_key = next(k for k in CID_KEYS if k in entry["test_case"])
            rec = {"set_index": i}
            try:
                conv = convert(entry, cid_key, meta["case_attributes"], meta["case_num_att"])
                if conv is None:
                    rec.update(status="unusable")
                    n_skip += 1
                else:
                    r = runner.run(conv)
                    c, l, y_norm, p_norm = r["preds"][0]
                    ex_days = sorted(float(x["total_time"]) / 1440.0
                                     for x in entry["examples"])
                    rec.update(status="ok", cid=c, pl=l, norm_days=conv["norm"],
                               ex_mean_days=float(np.mean(ex_days)),
                               ex_median_days=float(np.median(ex_days)),
                               y_true_days=conv["y_true_days"],
                               y_true_norm=y_norm, y_pred_norm=p_norm,
                               y_pred_days=p_norm * conv["norm"],
                               n_train_graphs=len(conv["train"]),
                               epochs=r["epochs"], train_loss=r["train_loss"])
                    n_ok += 1
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}")
                n_skip += 1
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (n_ok + n_skip) % 20 == 0:
                el = time.time() - t0
                print(f"[{a.log}] {n_ok + n_skip} set  ok={n_ok} skip={n_skip}  "
                      f"{el / max(n_ok + n_skip, 1):.1f} s/set  elapsed {el / 60:.1f} min",
                      flush=True)
    print(f"[{a.log}] finished: ok={n_ok} skip={n_skip} in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()

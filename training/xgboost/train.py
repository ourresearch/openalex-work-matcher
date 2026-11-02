"""Train the matcher's XGBoost ranker from the released training data, exactly as the shipped model was trained.

    pip install xgboost==2.1.3 pandas pyarrow
    python3 training/xgboost/train.py --features <dir with training_features.parquet and old_work_features.parquet> --out model.json

Inputs: the feature rows of every (record, candidate work) pair (release assets; they are what pipeline/sql/features.sql computes),
the old-work pick-one labels (data/training/old_works/judges/opus_pick_one.jsonl.gz) and the type veto (data/training/type_veto.csv).
Training rows = the 100K training draw's pairs in the `train` split (label 1 = the candidate Opus picked; an `unsure` record's
candidates are all 0) plus the old-work shortlists (Opus's pick = 1, every other candidate shown = 0). Early stopping on the `valid`
split. Pairs across a vetoed type boundary are never trained on. The feature order and category lists are fixed
(features.json): xgboost 2.1.3 stores no category-to-code map, so scoring must use the same lists.
With xgboost 2.1.3 this writes a booster byte-identical to the shipped one (compare the sha256 in models/MANIFEST.json)."""
import argparse
import gzip
import hashlib
import json
import os

import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
JOIN = ("same", "version_preprint")

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True, help="folder holding training_features.parquet and old_work_features.parquet")
ap.add_argument("--out", default="model.json")
a = ap.parse_args()

meta = json.load(open(os.path.join(HERE, "features.json")))
FEATS, CATS = meta["features"], meta["categories"]
veto = pd.read_csv(os.path.join(ROOT, "data", "training", "type_veto.csv"))
VETO = set(zip(veto.type_a, veto.type_b))
no_veto = lambda d: d[[(x, y) not in VETO for x, y in zip(d.type_a, d.type_b)]]

T = no_veto(pd.read_parquet(os.path.join(a.features, "training_features.parquet")))
L = no_veto(pd.read_parquet(os.path.join(a.features, "old_work_features.parquet")))
T = T[T.part == "train"].copy()           # the file also holds the gold set's pairs (part = gold), never trained on
T.loc[T.label_src == "unsure", "label"] = 0

pick, shown = {}, {}
with gzip.open(os.path.join(ROOT, "data", "training", "old_works", "judges", "opus_pick_one.jsonl.gz"), "rt") as f:
    for line in f:
        v = json.loads(line)
        if v["ok"]:
            ans, s = v["answer"], v["record_key"]
            wid = lambda w: w[1:] if w else None
            pick[s] = wid(v["shortlist"].get(ans.get("pick"))) if ans["label"] in JOIN else None
            shown[s] = {wid(w) for w in v["shortlist"].values()}
L["s"], L["c"] = L.seed_id.astype(str), L.cid.astype(str)
L = L[L.s.isin(shown)]
L = L[[c in shown[s] for s, c in zip(L.s, L.c)]].copy()
L["label"] = [int(pick[s] == c) for s, c in zip(L.s, L.c)]
L["split"] = "train"
print(f"training pairs: {len(T[T.split == 'train']):,} new-record + {len(L):,} old-work ({int(L.label.sum()):,} positive); "
      f"valid {len(T[T.split == 'valid']):,}", flush=True)


def X(d):
    x = d[FEATS].copy()
    for c, v in CATS.items():
        x[c] = pd.Categorical(x[c].fillna("NA").astype(str), categories=v)
    num = [c for c in FEATS if c not in CATS]
    x[num] = x[num].apply(pd.to_numeric, errors="coerce").astype("float32")
    return x


tr = pd.concat([T[T.split == "train"], L]).sort_values(["seed_id", "cid"])
va = T[T.split == "valid"]
m = xgb.XGBClassifier(n_estimators=4000, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, eval_metric="aucpr",
                      early_stopping_rounds=100, random_state=1668, n_jobs=-1)
m.fit(X(tr), tr.label.values, eval_set=[(X(va), va.label.values)], verbose=False)
m.get_booster().save_model(a.out)
print(f"best iteration {m.best_iteration} (trees used when scoring: {m.best_iteration + 1}); "
      f"{a.out} sha256 {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")

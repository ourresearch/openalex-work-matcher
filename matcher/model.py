"""Stage 1: hard rules, then XGBoost.

Hard rules drop a pair before it is scored: a vetoed type boundary (`models/type_veto.csv`: matching trusts the record's
type, so an article never joins a book chapter, a dataset, a dissertation...) and two undeclared Kaggle dataset versions
(both DOIs under 10.34740/kaggle/dsv/). XGBoost then scores every remaining pair; its raw score ranks the record's
candidates, and a record whose best score is under 0.01 is a new work.
"""
import csv
import json
import os

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
KAGGLE = "10.34740/kaggle/dsv/"
MINT_BELOW = 0.01


def load_veto(path=os.path.join(MODELS, "type_veto.csv")):
    with open(path) as f:
        return {(r["type_a"], r["type_b"]) for r in csv.DictReader(f)}


def vetoed(row, record_dois, work_dois, veto):
    if (row.get("type_a"), row.get("type_b")) in veto:
        return "type_veto"
    if any(d.startswith(KAGGLE) for d in record_dois) and any(d.startswith(KAGGLE) for d in work_dois):
        return "kaggle_versions"
    return None


class XGBoostScorer:
    """The released booster with its feature order and pinned category lists. xgboost 2.1 stores no category -> code map,
    so the categories must be given in exactly this order or scores are silently wrong."""

    def __init__(self, booster=os.path.join(MODELS, "xgboost.json"), meta=os.path.join(MODELS, "xgboost_meta.json")):
        import xgboost as xgb
        with open(meta) as f:
            m = json.load(f)
        self.feats, self.cats, self.n_trees = m["features"], m["categories"], m["n_trees"]
        self.booster = xgb.Booster(model_file=booster)

    def frame(self, rows):
        import pandas as pd
        x = pd.DataFrame([{c: r.get(c) for c in self.feats} for r in rows], columns=self.feats)
        for c, v in self.cats.items():
            x[c] = pd.Categorical(x[c].astype(object).where(x[c].notna(), "NA").astype(str), categories=v)
        num = [c for c in self.feats if c not in self.cats]
        x[num] = x[num].astype("float64").astype("float32")
        return x

    def score(self, rows):
        """Raw P(same work) for each feature row."""
        import xgboost as xgb
        if not rows:
            return []
        d = xgb.DMatrix(self.frame(rows), enable_categorical=True)
        return [float(p) for p in self.booster.predict(d, iteration_range=(0, self.n_trees))]

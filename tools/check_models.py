"""Check that the released models give the scores the benchmarks were computed with.

    pip install xgboost==2.1.3 pandas pyarrow
    python3 tools/download.py training_features.parquet --to assets/
    python3 tools/check_models.py xgboost --features assets/

xgboost: scores every gold (record, candidate) pair's feature row (training_features.parquet, part = gold, minus pairs across a
vetoed type boundary) with models/xgboost.json through matcher.model.XGBoostScorer, and compares each score with the shipping
XGBoost score cached in benchmarks/data/gold/xgboost.jsonl.gz (stored to 7 decimals). OK when every difference is under 1e-6."""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def check_xgboost(features_dir):
    import pandas as pd
    from matcher.model import XGBoostScorer, load_veto

    veto = load_veto()
    x = pd.read_parquet(os.path.join(features_dir, "training_features.parquet"))
    x = x[x.part == "gold"]
    x = x[[(a, b) not in veto for a, b in zip(x.type_a, x.type_b)]]
    scorer = XGBoostScorer()
    rows = x[scorer.feats].astype(object).where(x[scorer.feats].notna(), None).to_dict("records")
    got = scorer.score(rows)
    want = {}
    with gzip.open(os.path.join(ROOT, "benchmarks", "data", "gold", "xgboost.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["p"] is not None:
                want[(r["record_key"], r["work"])] = r["p"]
    keys = [(str(s), f"W{c}") for s, c in zip(x.seed_id, x.cid)]
    missing = [k for k in keys if k not in want]
    diffs = [abs(p - want[k]) for k, p in zip(keys, got) if k in want]
    worst = max(diffs) if diffs else float("nan")
    print(f"xgboost: {len(got):,} gold pairs scored, {len(diffs):,} compared with the cached scores "
          f"({len(missing):,} not cached, {len(want) - len(diffs):,} cached pairs not in the feature file)")
    print(f"max |diff| {worst:.2e}  {'OK' if diffs and not missing and len(diffs) == len(want) and worst < 1e-6 else 'MISMATCH'}")
    return worst < 1e-6 and not missing and len(diffs) == len(want)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["xgboost"])
    ap.add_argument("--features", required=True, help="folder holding training_features.parquet (a release asset)")
    a = ap.parse_args()
    sys.exit(0 if check_xgboost(a.features) else 1)

"""Qwen stage: the prompt format, shared by training and scoring (production builds the same string in pipeline/notebooks/
ScoreQwenModal.py and the matcher package). One row = the record and its shortlist (at most 10 candidates, in XGBoost rank
order) as JSON summaries, then one answer slot per candidate; train.py teaches ' yes' at the slot of the candidate Opus picked and
' no' at every other slot.

    python3 training/qwen/rows.py summaries.jsonl > rows_train.jsonl

`summaries.jsonl` holds one JSON object per line, {"key": <record_key or W id>, "summary": {...}}, for every record and
candidate work in data/training/qwen_rows.jsonl.gz. A summary has the fields the judges saw (training/prompts/README.md):
title, authors (first 8), n_authors, year, type, doi, venue, version, host, biblio, abstract (first 400 characters).
The repository does not redistribute titles or abstracts; tools/fetch_works.py rebuilds work summaries from the OpenAlex API.
Records and works change, so prompts rebuilt today are close to, not identical with, the ones the shipped adapter saw."""
import gzip
import json
import os
import sys

HEAD = ("Incoming bibliographic record and a shortlist of existing works. For each candidate answer yes only if the record "
        "is the same work (a copy, or the preprint / published version of it); otherwise no. At most one yes.\n")


def clean(summary):
    return {k: v for k, v in summary.items() if k not in ("prov", "cites") and v not in (None, "", [])}


def row(key, record, candidates):
    """candidates: [(work id, summary, y)] in rank order -> {seed_id, prefix, seats}"""
    lines = [f"RECORD: {json.dumps(clean(record), ensure_ascii=False)}", "CANDIDATES:"]
    lines += [f"c{i + 1}: {json.dumps(clean(c), ensure_ascii=False)}" for i, (_, c, _) in enumerate(candidates)]
    return {"seed_id": key, "prefix": HEAD + "\n".join(lines) + "\nANSWERS:\n",
            "seats": [{"line": f"c{i + 1}", "y": y, "key": w} for i, (w, _, y) in enumerate(candidates)]}


if __name__ == "__main__":
    summ = {}
    for line in open(sys.argv[1]):
        j = json.loads(line)
        summ[j["key"]] = j["summary"]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with gzip.open(os.path.join(root, "data", "training", "qwen_rows.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] != "train" or r["record_key"] not in summ:
                continue
            cands = [(s["work"], summ.get(s["work"], {}), s["y"]) for s in r["shortlist"]]
            print(json.dumps(row(r["record_key"], summ[r["record_key"]], cands), ensure_ascii=False))

# Reproduce

Every benchmark number comes from files in this repository: the models' scores and every judge's verdict are cached, so the
first two sections need no GPU, no API key and no private resource. The repository holds ids, not titles or abstracts;
[`tools/fetch_works.py`](tools/fetch_works.py) rebuilds text from the public OpenAlex API when a step needs it. Written for a
person or an AI agent: each command says what it should print.

You need Python 3.10 or later.

```bash
git clone https://github.com/ourresearch/openalex-work-matcher
cd openalex-work-matcher
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## The benchmarks (seconds, a laptop, no GPU or key)

```bash
python3 benchmarks/score.py all
```

recomputes every number in [benchmarks/README.md](benchmarks/README.md) from the cached scores and verdicts in `benchmarks/data/`, through
the same decision rule production runs (XGBoost cut at 0.01, Qwen bind at 0.910, copy sums). It should print:

```
== The matcher as it ships, on 8 judged nights of incoming records
  without copy sums            precision  99.1% (90% range 98.1%-99.7%)  found  91.0%  judged records 7,974, binds 4,105, with a match 4,522
  with copy sums (shipping)    precision  99.1% (90% range 98.1%-99.7%)  found  92.2%  judged records 7,974, binds 4,182, with a match 4,522
  by source of the incoming record:
    crossref   precision  99.2%  found  91.0%  (judged 1,883, binds 1,042)
    datacite   precision  99.7%  found  84.2%  (judged 1,890, binds 973)
    pubmed     precision  97.9%  found  94.9%  (judged 774, binds 331)
    repo       precision  99.0%  found  93.8%  (judged 3,427, binds 1,836)
    replay nights 18-22 Oct  precision  99.6%  found  91.8%
    live nights 26-28 Oct    precision  98.7%  found  92.5%
    titles of 3 words or fewer: precision  99.1%  found  69.1%  (judged 901)

== The title + author rule the matcher replaced, on the same judged records
  8 nights, same labels         precision  78.6%  found  74.3%  (binds 3,361)
  7 nights 16-22 Oct, Opus alone precision  76.9%  found  70.7%  (judged 7,244)

== Frozen gold: 3,359 records drawn from three nights of incoming records, every candidate pair labelled
  the shipping matcher (its XGBoost's 0.01 cut and top 10, then Qwen)
    without copy sums          precision  99.45%  found  93.5%  binds 1,031, wrong 6
    with copy sums (shipping)  precision  99.45%  found  94.2%  binds 1,044, wrong 6
  for reference: the earlier XGBoost's cut and top 10, the same Qwen
    without copy sums          precision  99.45%  found  93.4%  binds 1,032, wrong 5
    with copy sums (shipping)  precision  99.46%  found  94.3%  binds 1,044, wrong 5
  the two XGBoost models on these records: same 0.01 cut for 3,303 of 3,359, same top-10 set for 2,195

== Held-out records: incoming records never trained on whose best candidate scores 0.01 or more
  the shipping matcher (its XGBoost's cut and top 10, then Qwen): 9,154 records; threshold for 99% precision 0.904651 (production uses 0.910)
    without copy sums          precision  99.04%  found  93.9%  binds 7,392, wrong 71
    with copy sums (shipping)  precision  98.74%  found  94.5%  binds 7,461, wrong 94
     of the 23 binds copy sums add that the single-pick label calls wrong, 23 join a copy of the work Opus picked
  for reference: the earlier XGBoost's cut and top 10, where the threshold was set: 8,568 records; threshold for 99% precision 0.909907 (production uses 0.910)
    without copy sums          precision  99.04%  found  93.8%  binds 6,995, wrong 67
    with copy sums (shipping)  precision  98.69%  found  94.4%  binds 7,063, wrong 92
     of the 25 binds copy sums add that the single-pick label calls wrong, 24 join a copy of the work Opus picked
```

(explanatory lines under each heading omitted). `python3 benchmarks/score.py nights` (or `baseline`, `gold`, `heldout`) runs one
set. The scores are the shipping models' outputs, cached; to produce them yourself, score the same shortlists with the
released models (see "The released models are the production models" below).

## Retrain XGBoost (about a minute on a 14-core machine)

```bash
pip install xgboost==2.1.3 pandas pyarrow scikit-learn
python3 tools/download.py training_features.parquet old_work_features.parquet --to assets/
python3 training/xgboost/train.py --features assets/ --out model.json
```

trains on the released feature rows and labels with the shipped recipe and prints the booster's sha256. With xgboost 2.1.3 it
is byte-identical to the shipped model:

```
training pairs: 774,806 new-record + 111,641 old-work (3,588 positive); valid 114,321
best iteration 780 (trees used when scoring: 781); model.json sha256 85397f9eefae041e20d341d62f0b751f033e9309bf14d7e6565f366304b2aa45
```

## Retrain Qwen (about 2 hours on 8 H200 GPUs)

The training rows are published as ids (`data/training/qwen_rows.jsonl.gz`: each record's shortlist in rank order and the
yes / no target per candidate). Rebuild their text from the OpenAlex API, write the prompts with `training/qwen/rows.py`, then
run `training/qwen/train.py` (command in its docstring; `training/qwen/modal_app.py` runs it on Modal). Records and works have
changed since the adapter was trained, so a retrain is close to the released adapter, not identical to it.

## The released models are the production models

`models/xgboost.json` is the booster production loads (sha256 in [models/MANIFEST.json](models/MANIFEST.json)), and the
adapter in the `qwen-adapter.tar` release file is the one production serves. We checked both against a production night (the
2 November 2026 build, 2,031 records, 26,213 candidate pairs), on production's own inputs:

| Step | Repository code | Result |
|---|---|---|
| Features | `matcher/features.py` | all 95 features equal on every pair |
| Hard rules | `matcher/model.py` | the same 6,222 pairs dropped (6,189 type veto, 33 Kaggle) |
| XGBoost | `matcher/model.py` + `models/xgboost.json` | scores equal to 1.2e-7 (float32), the same rank order |
| Qwen prompts | `matcher/prompt.py` | all 2,019 identical, character for character |
| Qwen | `matcher/qwen.py` + the released adapter | all 17,149 scores identical, no decision changed |
| Embedding | `matcher/embed.py` | cosine scores within 0.0016 of production's index |

Those records are not redistributed (they carry titles and abstracts), so this table is a report, not a command. The XGBoost
half can be checked from released files: score the gold set's feature rows with the released booster and compare with the
cached scores.

```bash
python3 tools/download.py training_features.parquet --to assets/
python3 tools/check_models.py xgboost --features assets/
```

## Run the matcher on records

```bash
python3 tools/download.py qwen-adapter.tar --to assets/
export OPENALEX_API_KEY=...
python3 scripts/match.py examples/records.jsonl --adapter assets/qwen-adapter
```

On a GPU this prints one decision per example: the HAL copy of the NumPy paper joins W2146292423 (`qwen_bind`), the arXiv
preprint joins W3035965352 by its DOI (`identifier:doi`), and the made-up record is a new work (`xgb_mint`). Without a GPU,
`--stage1-only` prints the first two steps' answers and the third's Qwen prompt.

**Agreement with production.** We ran the tool on 400 records of the same production night, against the same works (the API
served that night's data), and compared with what production decided (one record hit an API timeout, which the client now handles). 397 of 399 records got the same decision: 383 joined the same work through Qwen
(one of them a copy-sum join in production), 3 through copy sums, and 10 became new works on both sides. In the 2 differences
the tool joined a work production left separate. The tool's best XGBoost candidate was production's for 396 records; its
search found 74% of production's candidates and 75% of the works on its top-10 shortlists. That night's comparison included
the works production had made for those records the same night, so most of them had an exact copy to find; matching records
as they arrive (below) is the harder test.

Run a record that is already in OpenAlex as if it were arriving with `--as-of <the date it arrived> --exclude-self`: works
created on or after that date and the record's own copy are left out.

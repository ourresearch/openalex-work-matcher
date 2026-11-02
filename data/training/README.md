# Training data

What the two models learned from. Ids only, as in [`benchmarks/`](../../benchmarks/README.md): a record is `provenance` +
`native_id_namespace` + `native_id`, a work is a W id, `record_key` joins the files. Licence: CC0.

## New records (both models)

100,000 records first seen on the nights of 10 to 12 October 2026, drawn by source × what the nightly did and weighted back to
those nights (`records.jsonl.gz`; none of them are in the gold set). Each record's candidates came from every search channel.
Jev scored every pair; the candidates at Jev 0.3 or more (top 10) plus the work the title rule bound went to Claude Opus 5.5 in
one pick-one call per record ([`judges/opus_pick_one.jsonl.gz`](judges/), 53,972 verdicts: the first pass and the re-picks
after later rulings; the last valid verdict per record counts).

`labels.jsonl.gz` has one row per (record, candidate) pair, 1,226,370 in all: `label` 1 = Opus's pick, 0 = not picked,
null = the record was `unsure`. `source` says why:

| source | pairs | label |
|---|---:|---|
| `same`, `version_preprint` | 28,032 | 1: Opus picked it |
| `shown_not_picked` | 127,357 | 0: on the shortlist, not picked |
| `not_shown` | 539,354 | 0: not on the shortlist (Jev under 0.3, or past the top 10) |
| `no_shortlist` | 492,969 | 0: the record had no candidate at Jev 0.3 or more |
| `unsure` | 30,356 | null: Opus could not decide; trained as 0 (a gold "indistinguishable" is a don't-bind too) |
| `veto_type`, `kaggle_undeclared` | 8,302 | 0: across a vetoed type boundary, or two undeclared Kaggle dataset versions |

Splits (`records.jsonl.gz`, `split`): train 71,385 records, valid 11,834 (early stopping), calib 11,393; valid + calib with a
candidate at 0.01 or more are the held-out benchmark. 5,388 records carry no split and were not trained on.

## Old works (the retrain)

To make the models work on records already in OpenAlex as well as new ones, a second draw of 21,491 works (one record each,
`old_works/records.jsonl.gz`, by stratum) went through the same candidate search and the XGBoost model of the time; 11,728 of
their shortlists were labelled by Opus pick-one (`old_works/judges/opus_pick_one.jsonl.gz`, final rubric). XGBoost trains on
those shortlists (pick = 1, every other candidate shown = 0); 10,006 of them join Qwen's training rows and 1,722 are a dev set
(with weights) for checking cutoffs on old works.

## Qwen's rows

`qwen_rows.jsonl.gz`: 49,556 rows, one per record: its shortlist in XGBoost rank order (at most 10) and the target per
candidate (`y` 1 on Opus's pick, 0 elsewhere; a record with no pick, `none` or `unsure`, is all 0: 24,520 of the 47,834
training rows). `set` says new record or old work; `split`
`train` (47,834 rows: 37,828 new records from the train split + 10,006 old works) or `dev`. The prompt text is not
redistributed; [`training/qwen/rows.py`](../../training/qwen/rows.py) writes it from record and work summaries.

## Type veto

`type_veto.csv`: the 104 (record type, work type) pairs that are never matched or trained on: article-family vs conference
paper / abstract, dissertation, report, book chapter, dataset, software, supplement; book vs chapter or review; notices and
peer reviews vs what they are about.

## Release assets

Too big for git; attached to the GitHub release and listed with sha256 in [`../RELEASE_ASSETS.json`](../RELEASE_ASSETS.json):

| File | Size | What |
|---|---:|---|
| `training_features.parquet` | 73 MB | the 95 features of every new-record training pair (`part` = train) and every gold pair (`part` = gold), with `label`, `label_src` and `split`: what `pipeline/sql/features.sql` computes |
| `old_work_features.parquet` | 11 MB | the same for every old-work candidate pair |
| `training_jev.jsonl.gz` | 37 MB | Jev's same-work score for every new-record training pair |

With these, [`training/xgboost/train.py`](../../training/xgboost/train.py) rebuilds the shipped XGBoost model byte for byte.

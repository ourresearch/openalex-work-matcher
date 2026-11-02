# Benchmarks

How often the matcher puts an incoming record on the right work, measured three ways, and how that compares with the title +
author rule it replaced. Every number below is recomputed from the files in [`data/`](data/) by one standard-library script:

```bash
python3 benchmarks/score.py all
```

## Results

| Set | What it is | Precision | Found |
|---|---|---:|---:|
| **Judged nights** | 7,974 real incoming records from 8 nights, judged by a panel | **99.1%** | **92.2%** |
| Frozen gold | 3,359 incoming records, every candidate pair labelled, disputes settled by a human reviewer | 99.45% | 94.2% |
| Held-out | 9,154 incoming records the models never saw; the bind threshold is set here | 99.04% | 93.9% |
| Title + author rule, judged nights | the same 7,974 records, the same labels | 78.6% | 74.3% |
| Title + author rule, first read-out | 7,244 judged records from 7 nights, Opus alone | 76.9% | 70.7% |

**Precision** is the share of binds that put the record on the right work. **Found** is the share of records that have a
matching work among their candidates that the matcher binds to it. A record that is not bound becomes a new work, so a miss
leaves a duplicate; a wrong bind merges two different works, which is worse, so the matcher is tuned for 99% precision.
Both are weighted so that the sample stands for every record of its night or frame (see each set below).

The held-out row is without copy sums (see below), because its labels can't grade them: with copy sums it reads 98.74% /
94.5%, and all 23 extra binds its labels call wrong join another copy of the paper the judge picked.

### On the judged nights, by where the record came from

| Source | Judged records | Precision | Found |
|---|---:|---:|---:|
| Crossref | 1,883 | 99.2% | 91.0% |
| DataCite | 1,890 | 99.7% | 84.2% |
| PubMed | 774 | 97.9% | 94.9% |
| Repositories | 3,427 | 99.0% | 93.8% |
| Titles of 3 words or fewer | 901 | 99.1% | 69.1% |

The 90% range of the nights' precision (400 resamples within sampling cells) is 98.1% to 99.7%. The live nights (26 to 28
October, the matcher running in shadow next to the nightly) read 98.7% / 92.5%; the replayed nights (18 to 22 October) 99.6% /
91.8%.

## How the matcher decides (what is scored)

For each incoming record that no identifier (DOI, PMID, arXiv id, a declared version or preprint link) already places:

1. XGBoost scores every candidate work. If the best is under 0.01, the record becomes a new work.
2. Qwen reads the record and XGBoost's top 10 together and gives each candidate a probability of being the same work.
3. The best candidate at **p ≥ 0.910** is bound.
4. **Copy sums:** otherwise, if the best candidate's p plus the p of the candidates that are copies of it (the same paper held
   by two works: the same title, a shared author name, years at most one apart) reaches 0.910, the best candidate is bound.
   Qwen was trained to say yes to one candidate, so it splits its yes between copies.
5. Otherwise the record becomes a new work.

Pairs across a vetoed type boundary ([`data/training/type_veto.csv`](../data/training/type_veto.csv): an article and a book
chapter, a dataset and an article, a notice and its article) are never scored.

## The sets

### Judged nights (`data/nights/`)

Each night from 16 to 28 October 2026 the matcher ran in shadow on every new record the nightly placed by title or minted,
and we drew a sample: every night's records split into cells by source × what the nightly did (bound by title, minted with a
DOI, minted without, parked) × how the matcher of the time compared with the nightly (both bind the same work, both mint,
the matcher binds, the matcher mints, different works), and up to a fixed number drawn per cell. A record's weight is
`weight × cell_n / cell_drawn` (`weight` is 1 on the live nights; on the replayed nights it carries the replay's own sampling).

**Labels.** Claude Opus 5.5 picked the right work from the record's shortlist, or none (pick-one prompt,
[`../training/prompts/`](../training/prompts/)). On records the matcher of the time bound (and, on the live nights, records it minted
where the nightly bound), a second judge (GPT-6.1 Sol) voted blind; where the two disagreed, Claude Fable 5.1 broke the tie (one
three-way split is left out). `unsure` counts as no match.

**Grading binds pair by pair.** A pick-one judge names one work, so a bind to a second copy of the same paper (a repository
record and its aggregator copy, both already in OpenAlex) would read as wrong. Every bind to a work other than the judges'
pick (216 pairs) was judged again as a pair, in both orders, by Opus with the pairwise prompt; it counts as right when both
orders say `same` or `version_preprint` ([`data/nights/judges/opus_pairwise.jsonl.gz`](data/nights/judges/)).

**Which nights.** The shipping models re-scored the 8 nights whose candidate works could still be rebuilt exactly as they stood
that night (18 to 22 and 26 to 28 October). 16 and 17 October are in the files and in the title-rule read-out only. The 30
October file holds a sample of repository records where the matcher disagreed with the nightly, drawn to study one failure; it
is not a random sample and is not reported.

### Frozen gold (`data/gold/`)

3,458 records first seen on the nights of 10 to 12 October 2026, at most 300 per source × nightly decision, weighted back to
those nights (3,359 had candidates). Each record's candidates came from every search channel, against the works as they stood
before the records arrived. Every candidate pair Jev scored at 0.3 or more, every title-rule bind, and a 10% audit of the rest
were judged by Opus in both orders with the pairwise prompt (the audit found no matches under 0.3, so unjudged pairs under 0.3
are labelled separate). A pair is a match when both orders say `same` or `version_preprint`; `unsure` in one order without a
match in the other is "indistinguishable" and labelled separate. 120 pairs where the judges split, or the rubric was not yet
settled, were labelled by a human reviewer, who may open the DOI or landing page; human labels override. The gold's records were
fixed before any model was trained and were never used for training or to set a threshold. `labels.jsonl.gz` says where each label came from
(`source`).

**Scored as production scores.** The shipping XGBoost cuts and ranks each record's candidates (`xgboost.jsonl.gz`, `p`) and
the released Qwen adapter scores the shipping top 10 (`qwen.jsonl.gz`). For reference, `score.py gold` also prints the numbers
on the shortlists of the XGBoost model that preceded the shipping one (`p_earlier`, `qwen_earlier.jsonl.gz`; same Qwen): 99.46%
/ 94.3%. The two XGBoost models make the same 0.01 cut for 3,303 of the 3,359 records but put a different set in the top 10
for 1,164, and Qwen reads the whole shortlist at once, so the shipping row is the one to quote.

### Held-out (`data/heldout/`)

The training draw is 100,000 records first seen on the nights of 10 to 12 October (none of them gold records); 94,612 of them
are split by record into train (71,385), valid (11,834) and calib (11,393). Held-out = the valid and calib records whose best candidate scores 0.01 or more
by the shipping XGBoost (9,154), with its top 10, scored by the released Qwen adapter (one record whose own text fills Qwen's
12,000-token prompt before any candidate fits is a new work, as in production: `over_cap.json`). Labels are Opus's pick-one
verdicts (the same kind as the training labels: yes on the pick, no on the rest, nothing for `unsure`).

**The threshold was set here,** before the shipping XGBoost existed: on the earlier XGBoost's cut and top 10 (8,568 records,
`p_earlier` and `qwen_earlier.jsonl.gz`), the lowest Qwen score at which binds are at least 99% right, keeping the most found,
is 0.909907, and production binds at 0.910 (99.04% / 93.8% there). The same search on the shipping shortlists lands a little
lower, at 0.904651; production keeps 0.910, which is slightly stricter. `score.py heldout` prints both.

### Title + author rule (`baseline`)

What the nightly did before the matcher: bind a record to the work that already holds its normalized title + first author's
name key (when the key is longer than 20 characters, at most three works hold it, and the record's Crossref or DataCite DOI
does not conflict with one of the same kind on the work), else mint. Scored on the judged nights
with the same labels and weights (`title_rule_work` in `records.jsonl.gz`). Its binds to a work other than the judges' pick
were not re-judged pair by pair, so a few of its "wrong" binds may be copies of the right paper.

## What "right" means

The judges and the human reviewer worked from one rubric ([`../training/prompts/rubric.md`](../training/prompts/rubric.md)). In
short, a record joins a work when it is the **same item** (a repository, aggregator, catalogue or legacy copy; the same item
typed differently; the same text deposited twice) or **the preprint, working paper, submitted or accepted manuscript of it**.
The rulings that settle the hard cases:

- **One work:** a translation or translated metadata of the same item within one journal or repository (SciELO's
  per-language DOIs, machine-translated abstracts); a preprint and its published version in any form (article, chapter,
  conference paper, report, book, thesis); when a preprint has both a conference version and a journal article, it joins the
  version it declares, else the article; declared versions of one item (a DataCite concept DOI and its version DOIs, `.vN`
  DOIs, Crossref-declared versions such as F1000Research v1 / v2, Cochrane review updates); print and ebook editions and a
  reissue of the same book.
- **Separate works:** another journal's publication under its own ISSN and DOI, in any language (duplicate publications,
  co-publications, translations in a translation journal); a conference abstract or paper and the journal article; a
  correction, retraction or expression-of-concern notice and its article; a book chapter and an article, even with the same
  title and author; a peer review and the item it reviews; a new edition; multi-part articles; a retracted article and its
  corrected republication; neighbours in a templated series (data uploads, catalogue items, photographs, the same study at two
  meetings); a dataset or software deposit declared a version of an article.
- **Types are trusted:** a record's `type` is taken as given. Pairs across the vetoed type boundaries are never matched; a
  wrong type is fixed in the type data, not at merge.

## Files

| Path | Rows | What |
|---|---:|---|
| `data/nights/records.jsonl.gz` | 10,996 | judged records: identity (`provenance`, `native_id_namespace`, `native_id`), night, source, the title rule's decision and work, sampling cell and weights |
| `data/nights/xgboost.jsonl.gz` | 90,065 | the shipping XGBoost's score for every candidate of the re-scored nights |
| `data/nights/qwen.jsonl.gz` | 50,019 | the shipping Qwen's p for each shortlist candidate, its rank, and which other candidates it is a copy of |
| `data/nights/judges/*.jsonl.gz` | 19,555 | every verdict: Opus, Sol and Fable pick-one, Opus pairwise re-checks |
| `data/gold/records.jsonl.gz` | 3,400 | gold records with stratum and weight |
| `data/gold/labels.jsonl.gz` | 45,555 | one label per (record, candidate) pair and where it came from |
| `data/gold/xgboost.jsonl.gz`, `qwen.jsonl.gz`, `over_cap.json` | 35,843 / 11,327 | both XGBoost models' scores; the released Qwen on the shipping top 10 |
| `data/gold/qwen_earlier.jsonl.gz` | 28,222 | the same Qwen on the earlier XGBoost's top 10 (reference) |
| `data/gold/judges/*.jsonl.gz` | 60,714 | Opus pairwise (both orders), Jev, the human labels, Opus pick-one on 250 records (a format check) |
| `data/gold/judged_pairs.jsonl.gz` | 7,028 | which pairs were sent to the judges and why |
| `data/heldout/records.jsonl.gz` | 9,316 | held-out records with weights, and which set each is in (shipping cut, earlier cut) |
| `data/heldout/xgboost.jsonl.gz`, `qwen.jsonl.gz`, `over_cap.json` | 94,304 / 78,516 | both XGBoost models' scores; the released Qwen on the shipping top 10, with Opus's pick |
| `data/heldout/qwen_earlier.jsonl.gz` | 77,415 | the same Qwen on the earlier XGBoost's top 10, where the threshold was set (reference) |

Ids only: a record is `provenance` + `native_id_namespace` + `native_id` (look it up at
`https://api.openalex.org/locations?filter=native_id:<id>`); a work is a W id. `record_key` is an opaque key the files join on.
No titles or abstracts are redistributed; the judges' short reasons are kept as they wrote them. A shortlist's candidate
letters (`c1`, `c2`, ...) map to works in each verdict's `shortlist`.

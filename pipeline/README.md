# The matcher as OpenAlex runs it

This folder is the production code, copied as deployed (the commit is in [SOURCE](SOURCE)): a Databricks job that runs
inside OpenAlex's nightly build, after the night's records are parsed and typed and before they are written to works.
Internal names (warehouse ids, the Modal workspace, volume and secret names, the owner's email) are replaced by
placeholders; nothing else changed. The Python package in [`../matcher/`](../matcher/) is a line-for-line port of the same
logic for one record at a time; on a production night's own inputs its features, XGBoost scores and Qwen prompts are
identical to these.

## The job (`jobs/map_work_ids.yaml`)

| Task | Code | What it does |
|---|---|---|
| `match_locations_by_identifier` | `notebooks/MatchLocationsByIdentifier.py`, `sql/match_locations_by_identifier.sql`, `sql/work_id_lookups.sql` | every record that needs a work tonight; the work already holding its DOI, PMID or arXiv id, or the one it declares itself a version or preprint of; the old title rule's pick, kept as an input |
| `prepare` | `notebooks/PrepareMatchInputs.py`, `sql/frame.sql`, `sql/seeds.sql` | the night's records for the matcher (first seen, no identifier match) and their record rows; pins the works version they are matched against |
| `keys` | `sql/keys.sql` | blocking keys of every work (normalized title, first-author key, MinHash bands, identity keys) |
| `embed` | `sql/embed.sql` | each record's embedding (Qwen3-Embedding-0.6B, the corpus recipe) |
| `candidates` | `sql/candidates.sql` | the id, title, author + year and MinHash channels |
| `knn` | `notebooks/KnnSearch.py` (`notebooks/ExportVectorShards.py`, `jobs/location_work_match_vectors.yaml` keep the GPU copy of the vectors) | exact nearest-neighbour search over every work's vector on a GPU cluster, top 20 |
| `features` | `sql/knn_rank.sql`, `sql/features.sql` | the 95 features of every (record, candidate) pair |
| `score_xgb` | `notebooks/ScoreXgboost.py` | hard rules, then XGBoost on every pair |
| `score_qwen` | `notebooks/ScoreQwenModal.py`, `sql/qwen_pairs.sql`, `notebooks/modal_location_matcher.py` | records with a candidate at 0.01 or more: their top 10 to Qwen3.5-9B + adapter on Modal |
| `write_decisions` | `notebooks/WriteDecisions.py`, `sql/decisions.sql` | copy sums, then one decision per record, and each record's shortlist kept |
| `assign_work_ids` | `notebooks/AssignWorkIds.py`, `sql/assign_work_ids.sql` | new works minted, every record pinned to its work |

`notebooks/lwm.py` holds the shared helpers (the SQL runner, the copy check). The SQL runs on a Databricks SQL warehouse; the
notebooks on serverless compute, except the kNN search (a GPU cluster).

## Tables

The SQL reads OpenAlex's internal tables: `openalex.works.openalex_works` (works, the same rows the
[snapshot](https://help.openalex.org/access/snapshot/) and API serve), `openalex.works.locations_w_types` (every record, the rows the
API's `/locations` serves), and the matcher's own working tables (`openalex.works.location_work_match_*`). To run it
elsewhere, point those names at the corresponding tables of a snapshot.

## No fallback

When any step cannot run in full (no GPUs, the Qwen service down, a record without a decision, a tripwire) the job fails and
the nightly build stops before any record is written, rather than deciding with less than the whole stack.

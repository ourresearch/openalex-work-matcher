# Training

How the two models were trained, as code you can rerun. The data is in [`data/training/`](../data/training/README.md); what
counts as a match is in [`prompts/rubric.md`](prompts/rubric.md).

| Folder | What | Reproduces |
|---|---|---|
| [`xgboost/`](xgboost/) | `train.py`: the ranker, from the released feature rows and labels; `features.json`: the fixed feature order and category lists | byte for byte (xgboost 2.1.3), about a minute on a desktop |
| [`qwen/`](qwen/) | `rows.py` (the prompt), `common.py` (slots and tokens), `train.py` (LoRA on Qwen/Qwen3.5-9B), `score.py`, `modal_app.py` | close, not exact: the prompt text has to be rebuilt from today's records |
| [`prompts/`](prompts/) | the judges' rubric and the two instructions, with the prompt ids every verdict row carries | |

## XGBoost

Gradient-boosted trees (`XGBClassifier`, 4,000 rounds at most, learning rate 0.05, depth 8, early stopping on the valid split's
average precision) over 95 features of a (record, candidate work) pair: which search channels found the candidate, the
embedding similarity, title / abstract / author / ORCID / identifier / DOI overlaps, numbers in the title, year, volume, issue,
page, language and publisher agreement, declared relations between the two DOIs, the work's sources and versions, and each
candidate's standing among the record's other candidates. Six are categorical (source, type and version of each side). The
shipped model stopped at 781 trees. It ranks the candidates and decides only the cut: a record whose best candidate scores
under 0.01 becomes a new work.

```bash
pip install xgboost==2.1.3 pandas pyarrow
python3 training/xgboost/train.py --features <folder with the two feature parquets> --out model.json
```

## Qwen

A LoRA adapter (rank 32, alpha 64, dropout 0.05, on every attention, linear-attention and MLP projection of the language model) on
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), trained 2 epochs at learning rate 1e-4 with an effective batch of
32 records on 8 GPUs (bf16, about 2 hours on H200s). One row is the record and its shortlist in one prompt
(`rows.py`), with an answer slot after each candidate; the loss is cross-entropy on ' yes' vs ' no' at each slot, yes only at
the candidate Opus picked. At scoring time p = softmax(' yes', ' no') at each slot, one forward pass per record, prompts capped
at 12,000 tokens (`score.py`, as production).

Rows: `data/training/qwen_rows.jsonl.gz`, split `train` (47,834). The threshold 0.910 was then set on the held-out records
(`benchmarks/score.py heldout`).

# Judge prompts

Every label in this repository that came from a model came from one of two prompts. Each is the rubric
([rubric.md](rubric.md)) followed by one instruction:

| Prompt | Instruction | Input (the user message, JSON) | Output |
|---|---|---|---|
| Pick one | [pick_one.txt](pick_one.txt) | `{"record": {...}, "candidates": [{"id": "c1", ...}, ...]}` | `{"label": same / version_preprint / none / unsure, "pick": "c3" or "none", "reason": ...}` |
| Pairwise | [pairwise.txt](pairwise.txt) | `{"record_a": {...}, "record_b": {...}}` | `{"label": same / version_preprint / version_other / different / unsure, "reason": ...}` |

The rubric is the system prompt; the record or pair is the only user message. A record is the same summary the matcher's
Qwen stage reads (`matcher` package, `summary()`): title, the first 8 authors and the author count, year, type, DOI, venue,
version, the landing page's host, volume / issue / first page, and the first 400 characters of the abstract, with empty
fields left out. In a pick-one call the candidates are shown in a fixed shuffled order, so the judge cannot learn from the
order the matcher ranked them in.

Mergeable = `same` or `version_preprint`. A pick-one `unsure` is a don't-bind: it counts as "no match" for training and for
grading. A pairwise pair is mergeable only when both orders (`ab`: the incoming record first, `ba`: swapped) say so.

## Versions

The rubric grew as cases were settled (each ruling in its header carries its date). Every verdict row in `benchmarks/data/` and
`data/training/` names the prompt it was made with: the first 8 hex digits of the SHA-1 of the full system prompt.

| Prompt id | Kind | Rubric as of | Used for |
|---|---|---|---|
| `3e7af1f0` | pairwise | 2026-10-12 | the frozen gold's first pass (both orders, every selected pair) |
| `a48fbd02` | pairwise | 2026-10-13 | the gold's re-judge after the first rulings (169 pairs) |
| `cb28ae19` | pick one | 2026-10-13 | the 100K training draw; the 250-record format check on the gold |
| `74fcdd8d` | pick one | 2026-10-15 | re-picks of training records a 10-15 ruling touched |
| `f7e5b633` | pick one | 2026-10-17 (final) | the judged nights (all three judges); the old-work labels |
| `f9cccc47` | pairwise | 2026-10-17 (final) | the judged nights' pair-by-pair check of binds to a work other than the judges' pick |

A few rows carry other ids (`71f3732a`, `8f12d710`, `e70d693b`, `576cedf8`, `b877eb37`): targeted re-judges run while a
ruling was being written up (a book re-judge, the embedding-candidate pairs). Their rubric text was not kept byte for byte;
it sits between the dated versions above.

**The published text does not hash to these ids.** The rubric as sent also named the people who made each ruling and pointed
to internal notes; [rubric.md](rubric.md) is the final version (`f7e5b633` / `f9cccc47`) with only those references removed or
reworded. The rules themselves are unchanged.

## Models

| Judge | Model | Where it was used |
|---|---|---|
| Opus | Claude Opus 5.5 | every pick-one and pairwise label |
| Sol | GPT-6.1 Sol, reasoning effort xhigh | second vote on the judged nights' model binds and model mints |
| Fable | Claude Fable 5.1, reasoning effort xhigh | tie-break where Opus and Sol disagreed on a judged night |
| Jev | Jev, a decision model from [TypeSafe AI](https://typesafe.ai) | a same-work score on every candidate pair; pairs under 0.3 were not sent to Opus (a 10% audit found no merges below it) |
| Human | a human reviewer | 120 gold pairs where the judges disagreed or the rubric was unsettled; may open the DOI or landing page |

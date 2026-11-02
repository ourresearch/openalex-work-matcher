"""Match records to OpenAlex works, the way OpenAlex matches its incoming records.

    export OPENALEX_API_KEY=...
    python3 scripts/match.py records.jsonl --adapter assets/qwen-adapter > decisions.jsonl

records.jsonl: one record per line, shaped like an OpenAlex location (README "Use it"). Each output line is the record's key
fields plus the decision: `work` (an OpenAlex work id, or null = a new work), `layer` (which step decided), XGBoost's best
candidate and score, Qwen's best candidate and score, and the shortlist Qwen read.

--stage1-only runs everything but Qwen (no GPU): records whose best XGBoost score is under 0.01 get their final answer
(a new work), the rest are written with their Qwen prompt for scoring elsewhere (matcher.qwen).
--as-of DATE and --exclude-self match a record that is already in OpenAlex as if it were arriving on DATE: works created on or
after DATE and the record's own location are left out.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matcher.match import Matcher  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("records")
    p.add_argument("--adapter", help="the unpacked Qwen adapter folder (tools/download.py qwen-adapter.tar)")
    p.add_argument("--stage1-only", action="store_true")
    p.add_argument("--as-of")
    p.add_argument("--exclude-self", action="store_true")
    p.add_argument("--no-identifiers", action="store_true", help="skip stage 0 (DOI / PMID / declared versions)")
    p.add_argument("--device")
    a = p.parse_args()
    if not a.stage1_only and not a.adapter:
        p.error("--adapter is required unless --stage1-only")
    m = Matcher(adapter=a.adapter, device=a.device, load_qwen=not a.stage1_only)
    kw = {"as_of": a.as_of, "exclude_self": a.exclude_self, "identifiers": not a.no_identifiers}
    for line in open(a.records):
        rec = json.loads(line)
        key = {k: rec.get(k) for k in ("provenance", "native_id_namespace", "native_id") if k in rec}
        try:
            if a.stage1_only:
                s1 = m.stage1(rec, **kw)
                out = {**key, **(s1["decision"] if s1["done"] else {"layer": "needs_qwen", "prompt": s1["prompt"],
                                                                     "xgb": s1["xgb"][:10]})}
            else:
                out = {**key, **m.match(rec, **kw)}
        except Exception as e:   # one bad record does not stop the file; production instead fails the night
            out = {**key, "error": repr(e)[:500]}
        print(json.dumps(out, ensure_ascii=False), flush=True)
    print(f"{m.api.calls} API calls", file=sys.stderr)


if __name__ == "__main__":
    main()

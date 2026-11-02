"""Rebuild the summaries the judges and Qwen read, from the public OpenAlex API, for the records and works a data file names
(the repository ships ids only).

    python3 tools/fetch_works.py data/training/qwen_rows.jsonl.gz > summaries.jsonl
    python3 training/qwen/rows.py summaries.jsonl > rows_train.jsonl

Output: one {"key": <record_key | W id>, "summary": {...}} per line. A work's summary is its current API record. A record's
summary comes from its location row and the work it sits on today (the API serves records without their author list, date,
pages or abstract). Records and works keep changing, so rebuilt prompts are close to, not identical with, the ones the models
were trained and tested on. Needs OPENALEX_API_KEY."""
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matcher import openalex as OA, prompt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def records_index():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "**", "records.jsonl.gz"), recursive=True):
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            out.setdefault(r["record_key"], r)
    return out


def main(path):
    rec_keys, work_ids = set(), set()
    for l in (gzip.open(path, "rt") if path.endswith(".gz") else open(path)):
        j = json.loads(l)
        if "record_key" in j:
            rec_keys.add(j["record_key"])
        for s in j.get("shortlist") or []:
            work_ids.add(OA.wid(s["work"]))
        for k in ("work", "cid", "pick"):
            if j.get(k):
                work_ids.add(OA.wid(j[k]))
    api, recs = OA.OpenAlex(), records_index()
    rec_loc = {}
    for k in sorted(rec_keys):
        r = recs.get(k)
        if not r:
            continue
        ls = api.locations(f"provenance:{r['provenance']},native_id:{r['native_id']}")
        if ls:
            rec_loc[k] = ls[0]
            if ls[0].get("work_id"):
                work_ids.add(OA.wid(ls[0]["work_id"]))
    aw, al = api.works_by_id(work_ids), api.locations_of(work_ids)
    works = {i: OA.work_from_api(w, al.get(i, [])) for i, w in aw.items()}
    for i, w in works.items():
        print(json.dumps({"key": f"W{i}", "summary": prompt.work_summary(w)}, ensure_ascii=False))
    for k, l in rec_loc.items():
        w = works.get(OA.wid(l["work_id"])) if l.get("work_id") else None
        rec = {"title": l.get("title"), "type": l.get("type"), "best_doi": OA._bare_doi(l.get("doi")), "source_name": l.get("source_name"),
               "version": l.get("version"), "landing_page_url": l.get("landing_page_url"), "provenance": l.get("provenance"),
               "authors": [{"name": a.get("display_name")} for a in (w or {}).get("authorships") or []] if w else None,
               "year": (w or {}).get("publication_year"), "abstract": (w or {}).get("abstract")}
        b = (w or {}).get("biblio") or {}
        rec.update(volume=b.get("volume"), issue=b.get("issue"), first_page=b.get("first_page"))
        print(json.dumps({"key": k, "summary": prompt.record_summary(rec)}, ensure_ascii=False))
    print(f"{len(works)} works, {len(rec_loc)} of {len(rec_keys)} records; {api.calls} API calls", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])

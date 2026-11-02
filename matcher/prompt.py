"""Stage 2 input: the record and its top 10 candidates by XGBoost, in the summary format the Qwen model was trained on
(`pipeline/sql/qwen_pairs.sql` and the prompt builder of the production Qwen step). The model answers " yes" or " no" after
each candidate line; its P(yes) is the stage 2 score."""
import json
import re

from .features import year

HEAD = ("Incoming bibliographic record and a shortlist of existing works. For each candidate answer yes only if the record "
        "is the same work (a copy, or the preprint / published version of it); otherwise no. At most one yes.\n")
MAX_CANDIDATES = 10
HOST = re.compile(r"^https?://([^/]+)")


def _host(url):
    if url is None:
        return None
    m = HOST.search(url)
    return m.group(1) if m else ""


def _ws(*xs):
    """concat_ws(' ', ...): NULLs skipped, empty strings kept."""
    return " ".join(x for x in xs if x is not None)


def _clean(d, drop=("prov", "cites")):
    return {k: v for k, v in d.items() if k not in drop and v is not None and v != "" and v != []}


def record_summary(rec):
    authors = rec.get("authors")
    if authors is not None:
        authors = [(a or {}).get("name") or _ws((a or {}).get("given"), (a or {}).get("family")) for a in authors[:8]]
    b = rec.get("abstract")
    return {"title": rec.get("title"), "authors": authors, "n_authors": None if rec.get("authors") is None else len(rec["authors"]),
            "year": year(rec.get("published_date", rec.get("year"))), "type": rec.get("type"), "doi": rec.get("best_doi"),
            "venue": rec.get("source_name"), "version": rec.get("version"), "host": _host(rec.get("landing_page_url")),
            "biblio": _ws(rec.get("volume"), rec.get("issue"), rec.get("first_page")),
            "abstract": None if b is None else b[:400], "prov": rec.get("provenance")}


def work_summary(w):
    auths = w.get("authorships")
    pl = w.get("primary_location") or {}
    bib = w.get("biblio") or {}
    b = w.get("abstract")
    return {"title": w.get("title"), "authors": None if auths is None else [(a or {}).get("display_name") for a in auths[:8]],
            "n_authors": None if auths is None else w.get("n_authorships", len(auths)), "year": w.get("publication_year"),
            "type": w.get("type"), "doi": w.get("doi"), "venue": pl.get("source_name") if pl.get("source_name") is not None else pl.get("raw_source_name"),
            "version": pl.get("version"), "host": _host(pl.get("landing_page_url")),
            "biblio": _ws(bib.get("volume"), bib.get("issue"), bib.get("first_page")),
            "abstract": None if b is None else b[:400], "prov": pl.get("provenance"), "cites": w.get("cited_by_count")}


def build(rec, shortlist):
    """shortlist: [(cid, work)] in XGBoost rank order. Returns {"prefix": str, "seats": [{"line", "key"}]}."""
    cs = shortlist[:MAX_CANDIDATES]
    lines = [f"RECORD: {json.dumps(_clean(record_summary(rec)), ensure_ascii=False)}", "CANDIDATES:"]
    lines += [f"c{i + 1}: {json.dumps(_clean(work_summary(w)), ensure_ascii=False)}" for i, (_, w) in enumerate(cs)]
    return {"prefix": HEAD + "\n".join(lines) + "\nANSWERS:\n",
            "seats": [{"line": f"c{i + 1}", "key": str(cid)} for i, (cid, _) in enumerate(cs)]}

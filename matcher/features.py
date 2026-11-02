"""Model features for one incoming record and its candidate works: a line-for-line port of the production feature SQL
(`pipeline/sql/features.sql`), including its NULL behaviour, so the released XGBoost model sees exactly what it saw in
training.

A record is a dict shaped like an OpenAlex location (see README "Use it"); a work is the internal work dict built by
`matcher.openalex.work_from_api` (or read from the snapshot). Each candidate is a dict
{"cid": int, "work": work, "channels": [..], "emb_rank": int | None, "emb_score": float | None, "rel": set(..)}.
"""
import re

from . import text

DP_KEY = re.compile(r"^d:(10\.[0-9]+)/")
DP_ANY = re.compile(r"(10\.[0-9]+)/")
SOURCE_NUM = re.compile(r"([0-9]+)$")
REL_CLASSES = ("version", "sibling", "identical", "part", "supplement", "derived", "mention", "other")


def year(d):
    if d is None:
        return None
    if isinstance(d, int):
        return d
    m = re.match(r"^\s*(\d{4})", str(d))
    return int(m.group(1)) if m else None


def _exists(values, pred):
    """Spark exists() with three-valued logic: true if any true, else NULL if any NULL, else false."""
    if values is None:
        return None
    seen_null = False
    for v in values:
        r = pred(v)
        if r is True:
            return 1
        if r is None:
            seen_null = True
    return None if seen_null else 0


def _eq(a, b):
    return None if a is None or b is None else int(a == b)


def _agree(a, b):
    return 0 if a is None or b is None else (1 if a == b else -1)


def _add(d, kind, t):
    if t is not None and t != "":
        d.setdefault(kind, set()).add(t)


def _doi_tokens(d, keys):
    for k in keys:
        if k.startswith("d:"):
            _add(d, "do", k[2:])
            m = DP_KEY.search(k)
            _add(d, "dp", m.group(1) if m else "")


def record_side(rec):
    """The record's half of every pair: its keys, scalar fields and token sets."""
    nt = text.norm_title(rec.get("title"))
    keys = text.record_keys(rec)
    authors = rec.get("authors")
    a100 = (authors or [])[:100]
    ids = rec.get("ids") or []
    prov = rec.get("provenance")
    tok = {}
    for x in (nt or "").split(" "):
        if len(x) >= 2:
            _add(tok, "ti", x)
        if re.search(r"[0-9]", x):
            _add(tok, "nu", x)
    for x in text.ln_tokens((rec.get("abstract") or "")[:2000].lower()):
        if len(x) >= 4:
            _add(tok, "ab", x)
    for a in a100:
        a = a or {}
        _add(tok, "su", text.surname(a.get("name"), a.get("family")))
        _add(tok, "oc", text.orcid(a.get("orcid")))
    for k in keys:
        _add(tok, "ky", k)
    _doi_tokens(tok, keys)
    if prov in ("crossref", "pubmed", "datacite"):
        for k in keys:
            if k.startswith("d:"):
                _add(tok, "dd" if prov == "datacite" else "dc", k[2:])
    for i in ids:
        if (i.get("namespace") or "").lower() == "doi" and i.get("id") is not None:
            d = text.bare_doi(i["id"])
            _add(tok, "do", d)
            if prov == "pubmed":
                _add(tok, "dc", d)
            m = DP_ANY.search(text.strim(i["id"]).lower())
            _add(tok, "dp", m.group(1) if m else "")
    if rec.get("source_id") is not None:
        _add(tok, "so", str(rec["source_id"]))
    pm = [i for i in ids if i.get("namespace") == "pmid"]
    pmid = None
    if pm and pm[0].get("id") is not None:
        m = re.search(r"([0-9]+)$", pm[0]["id"])
        pmid = m.group(1) if m else ""
    pub = rec.get("publisher")
    return {
        "nt": nt, "akey": text.author_key(text.record_a0(authors)), "keys": keys, "tok": tok,
        "provenance": prov, "type": rec.get("type"), "raw_type": rec.get("raw_type"), "version": rec.get("version"),
        "language": rec.get("language"), "publisher": pub.lower() if pub is not None else None,
        "yr": year(rec.get("published_date", rec.get("year"))), "volume": text.nullif_trim(rec.get("volume")),
        "issue": text.nullif_trim(rec.get("issue")), "first_page": text.nullif_trim(rec.get("first_page")),
        "is_retracted": bool(rec.get("is_retracted") or False),
        "n_auth": None if authors is None else len(authors), "abs_len": len(rec.get("abstract") or ""),
        "title_len": None if rec.get("title") is None else len(rec["title"]), "pmid": pmid,
    }


def work_side(w):
    """The work's half of every pair."""
    nt = text.norm_title(w.get("title"))
    keys = text.work_keys(w)
    auths = w.get("authorships")
    a100 = (auths or [])[:100]
    locs = w.get("locations")
    tok = {}
    for x in (nt or "").split(" "):
        if len(x) >= 2:
            _add(tok, "ti", x)
        if re.search(r"[0-9]", x):
            _add(tok, "nu", x)
    for x in text.ln_tokens((w.get("abstract") or "")[:2000].lower()):
        if len(x) >= 4:
            _add(tok, "ab", x)
    for a in a100:
        a = a or {}
        _add(tok, "su", text.surname(a.get("display_name")))
        _add(tok, "oc", text.orcid(a.get("orcid")))
    for k in keys:
        _add(tok, "ky", k)
    _doi_tokens(tok, keys)
    for l in locs or []:
        if l.get("provenance") in ("crossref", "pubmed", "datacite"):
            _add(tok, "dd" if l["provenance"] == "datacite" else "dc", text.bare_doi(l.get("doi"), trim=False))
        sid = l.get("source_id")
        if sid is not None:
            m = SOURCE_NUM.search(sid)
            _add(tok, "so", m.group(1) if m else "")
    pmid = (w.get("ids") or {}).get("pmid")
    if pmid is not None:
        m = re.search(r"([0-9]+)$", pmid)
        pmid = m.group(1) if m else ""
    b = w.get("biblio") or {}
    pl = w.get("primary_location") or {}
    pub = w.get("publisher")
    first_author = (auths[0] or {}).get("display_name") if auths else None
    return {
        "nt": nt, "akey": text.author_key(first_author), "keys": keys, "tok": tok,
        "yr": w.get("publication_year"), "type": w.get("type"), "language": w.get("language"),
        "publisher": pub.lower() if pub is not None else None,
        "volume": text.nullif_trim(b.get("volume")), "issue": text.nullif_trim(b.get("issue")),
        "first_page": text.nullif_trim(b.get("first_page")),
        "is_retracted": w.get("is_retracted"), "is_xpac": w.get("is_xpac"),
        "n_auth": None if auths is None else w.get("n_authorships", len(auths)), "abs_len": len(w.get("abstract") or ""),
        "title_len": None if w.get("title") is None else len(w["title"]), "pmid": pmid,
        "locations_count": w.get("locations_count"), "cited_by_count": w.get("cited_by_count"),
        "w_crossref": _exists(locs, lambda l: None if l.get("provenance") is None else l["provenance"] == "crossref"),
        "w_datacite": _exists(locs, lambda l: None if l.get("provenance") is None else l["provenance"] == "datacite"),
        "w_pubmed": _exists(locs, lambda l: None if l.get("provenance") is None else l["provenance"] == "pubmed"),
        "w_repo": _exists(locs, lambda l: None if l.get("provenance") is None else l["provenance"] in ("repo", "pmh")),
        "w_published": _exists(locs, lambda l: None if l.get("version") is None else l["version"] == "publishedVersion"),
        "w_submitted": _exists(locs, lambda l: None if l.get("version") is None else l["version"] == "submittedVersion"),
        "w_accepted": _exists(locs, lambda l: None if l.get("version") is None else l["version"] == "acceptedVersion"),
        "prov": pl.get("provenance"), "version": pl.get("version"),
    }


def _ratio(a, b):
    return a / max(b, 1)


def pair_features(r, w, cand, bound_work, nt_fanout):
    """One feature row (no within-record features yet)."""
    rt, wt = r["tok"], w["tok"]
    n = lambda t, k: len(t.get(k, ()))
    inter = lambda k: len(rt.get(k, set()) & wt.get(k, set()))
    ch = cand.get("channels") or []
    rel = cand.get("rel") or set()
    f = {
        "ch_id": int("id" in ch), "ch_ti": int("ti" in ch), "ch_ay": int("ay" in ch), "ch_mh": int("mh" in ch),
        "ch_emb": int("emb" in ch), "is_bound": None if bound_work is None else int(cand["cid"] == bound_work),
        "emb_rank": cand.get("emb_rank"), "emb_score": cand.get("emb_score"),
        "same_nt": _eq(r["nt"], w["nt"]), "same_akey": _eq(r["akey"], w["akey"]),
        "nt_fanout": nt_fanout, "title_len_a": r["title_len"], "title_len_b": w["title_len"],
    }
    for k in ("ti", "ab", "su", "oc", "ky", "do", "nu"):
        f[f"{k}_na"], f[f"{k}_nb"], f[f"{k}_inter"] = n(rt, k), n(wt, k), inter(k)
    f["nu_mismatch"] = int(f["nu_na"] + f["nu_nb"] > 2 * f["nu_inter"])
    dc = n(rt, "dc") > 0 and n(wt, "dc") > 0 and inter("dc") == 0
    dd = n(rt, "dd") > 0 and n(wt, "dd") > 0 and inter("dd") == 0
    f["agency_doi_conflict"], f["dc_conflict"], f["dd_conflict"] = int(dc or dd), int(dc), int(dd)
    f["agency_dois_b"] = n(wt, "dd") if n(rt, "dd") > 0 else n(wt, "dc")
    f["dp_inter"] = inter("dp")
    f["so_na"], f["so_nb"], f["so_inter"] = n(rt, "so"), n(wt, "so"), inter("so")
    f["n_auth_a"], f["n_auth_b"], f["abs_len_a"], f["abs_len_b"] = r["n_auth"], w["n_auth"], r["abs_len"], w["abs_len"]
    f["yr_diff"] = None if w["yr"] is None or r["yr"] is None else w["yr"] - r["yr"]
    f["undated_a"], f["undated_b"] = int(r["yr"] is None), int(w["yr"] is None)
    f["pmid_agree"] = (1 if r["pmid"] == w["pmid"] else -1) if r["pmid"] and w["pmid"] else 0
    for k in ("volume", "issue", "first_page", "language", "publisher"):
        name = {"volume": "vol", "first_page": "page", "language": "lang"}.get(k, k)
        f[f"{name}_agree"] = _agree(r[k], w[k])
    f.update({"prov_a": r["provenance"], "type_a": r["type"], "type_b": w["type"], "version_a": r["version"],
              "prov_b": w["prov"], "version_b": w["version"], "raw_type_a": r["raw_type"],
              "retracted_a": int(r["is_retracted"]),
              "retracted_b": None if w["is_retracted"] is None else int(w["is_retracted"]),
              "xpac_b": None if w["is_xpac"] is None else int(w["is_xpac"]),
              "locations_count": w["locations_count"], "cited_by_count": w["cited_by_count"]})
    for k in ("w_crossref", "w_datacite", "w_pubmed", "w_repo", "w_published", "w_submitted", "w_accepted"):
        f[k] = w[k]
    for c in REL_CLASSES:   # NULL, not 0, for a pair with no declared relation of any class
        f[f"rel_{c}"] = int(c in rel) if rel else None
    for k in ("ti", "ab", "su"):
        i, a, b = f[f"{k}_inter"], f[f"{k}_na"], f[f"{k}_nb"]
        f[f"{k}_jacc"] = _ratio(i, a + b - i)
        f[f"{k}_cont"] = _ratio(i, min(a, b))
    return f


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def within_record(rows):
    """Window features over all of one record's pairs: count, gaps to the best candidate, rank by title overlap."""
    mx = {k: _max(r[k] for r in rows) for k in ("ti_jacc", "ab_jacc", "su_jacc", "emb_score")}
    same = [r["same_nt"] for r in rows if r["same_nt"] is not None]
    for r in rows:
        r["n_cands"] = len(rows)
        for k in ("ti_jacc", "ab_jacc", "su_jacc", "emb_score"):
            r[f"{k}_gap"] = None if r[k] is None or mx[k] is None else r[k] - mx[k]
        r["n_same_nt"] = sum(same) if same else None
        r["ti_jacc_rank"] = 1 + sum(1 for o in rows if o["ti_jacc"] > r["ti_jacc"])
    return rows


def features(rec, cands, bound_work=None, nt_fanout=None):
    """All feature rows for one record: [{"cid": .., **features}] in the order of `cands`."""
    r = record_side(rec)
    rows = []
    for c in cands:
        f = pair_features(r, work_side(c["work"]), c, bound_work, nt_fanout)
        f["cid"] = c["cid"]
        rows.append(f)
    return within_record(rows)

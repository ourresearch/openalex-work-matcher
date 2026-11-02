"""Stage 0: identifiers decide before any model (`pipeline/sql/match_locations_by_identifier.sql`), in this order:

  doi                the work that already holds the record's DOI
  pmid / arxiv       the work that already holds its PMID / arXiv id
  version_of         a DataCite record declaring IsVersionOf exactly one DOI (or a `.vN` DOI): the work holding that DOI
  has_version        a DataCite concept record declaring HasVersion exactly one DOI: the work holding that DOI
  crossref_preprint  a Crossref record declaring is-preprint-of / has-preprint: the work holding its partner DOI(s), when
                     they all sit on one work

A declared version binds on the declaration alone when both DOIs have the same prefix (one registrant); across registrants
the record's title + author key must also be on the target DOI's records. Production looks keys up in its map of every record
ever matched; here the API's /locations and /works answer the same questions for the records OpenAlex holds today.
"""
import re

from . import openalex as OA, titlekey

CLEAN = re.compile(r"[^a-zA-Z0-9./-]")
DOI_HEAD = re.compile(r"^(https?://(dx[.])?doi[.]org/|doi:)")
IDENTITY = {"self", "none", "isidenticalto", "isvariantformof", "isoriginalformof", "isversionof", "hasversion", "isnewversionof",
            "ispreviousversionof"}


def _clean(d):
    return CLEAN.sub("", DOI_HEAD.sub("", d.strip().lower())) if d else None


def _declared(rec, relationship):
    out = {_clean(i["id"]) for i in rec.get("ids") or [] if i.get("relationship") == relationship and i.get("id")
           and (i.get("namespace") or "doi").lower() == "doi" and "10.48550/" not in i["id"].lower()}
    return sorted(x for x in out if x)


class Identifiers:
    def __init__(self, api=None):
        self.api = api or OA.OpenAlex()

    def _doi_works(self, doi, rec, as_of, exclude_self):
        """{work id: [title keys on that DOI's records]}"""
        out = {}
        for l in self.api.locations(f"doi:{doi}"):
            if not l.get("work_id") or (exclude_self and OA.same_location(l, rec)):
                continue
            out.setdefault(OA.wid(l["work_id"]), []).append((l.get("merge_key") or {}).get("title_author"))
        if as_of and out:
            got = self.api.works_by_id(out)
            out = {i: v for i, v in out.items() if i in got and (got[i].get("created_date") or "")[:10] < as_of}
        return out

    def match(self, rec, as_of=None, exclude_self=False):
        """-> (work id, source) or (None, None)"""
        ids = rec.get("ids") or []
        doi = next((i["id"] for i in ids if i.get("namespace") == "doi" and i.get("id")), None)
        pmid = next((i["id"] for i in ids if i.get("namespace") == "pmid" and i.get("id")
                     and (i.get("relationship") or "none").lower() in IDENTITY), None)
        dc = _clean(doi)
        if dc:
            w = self._doi_works(dc, rec, as_of, exclude_self)
            if w:
                return min(w), "doi"
        if pmid:
            m = re.search(r"(\d+)$", pmid)
            if m:
                rs, _ = self.api.works(filter=f"pmid:{m.group(1)}")
                rs = [r for r in rs if not as_of or (r.get("created_date") or "")[:10] < as_of]
                if rs and not (exclude_self and rec.get("provenance") == "pubmed"):   # a PubMed record's own PMID is itself
                    return min(OA.wid(r["id"]) for r in rs), "pmid"
        arxiv = next((i["id"] for i in ids if i.get("namespace") == "arxiv" and i.get("id")), None)
        if arxiv:
            w = self._doi_works(_clean("10.48550/arxiv." + arxiv.split(":")[-1]), rec, as_of, exclude_self)
            if w:
                return min(w), "arxiv"
        key = titlekey.title_author(rec)
        gate = lambda target, keys: (dc and dc.split("/")[0] == target.split("/")[0]) or (
            key and len(key) > 20 and key in keys)
        if rec.get("provenance") == "datacite":
            vo = _declared(rec, "IsVersionOf")
            target = vo[0] if len(vo) == 1 else (re.sub(r"[.][vV][0-9]+$", "", dc) if dc and re.search(r"[.][vV][0-9]+$", dc) else None)
            for rel, t in (("version_of", target), ("has_version", (_declared(rec, "HasVersion") + [None])[0]
                                                    if len(_declared(rec, "HasVersion")) == 1 else None)):
                if t:
                    w = self._doi_works(t, rec, as_of, exclude_self)
                    if w:
                        i = min(w)
                        if gate(t, w[i]):
                            return i, rel
        if rec.get("provenance") == "crossref":
            partners = sorted({_clean(i["id"]) for i in ids if i.get("relationship") in ("is-preprint-of", "has-preprint") and i.get("id")})
            hits = {}
            for p in partners:
                for i in self._doi_works(p, rec, as_of, exclude_self):
                    hits.setdefault(p, set()).add(i)
            on = {min(v) for v in hits.values()}
            if len(on) == 1:
                return on.pop(), "crossref_preprint"
        return None, None

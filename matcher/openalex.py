"""A small OpenAlex API client and the adapters from API objects to the work dicts the matcher reads.

Set OPENALEX_API_KEY (a free key: https://openalex.org/settings/api). Every list query passes include_xpac=true: the
matcher's candidates include works that list endpoints hide by default.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.openalex.org"
WORK_FIELDS = ("id,doi,title,publication_year,type,language,authorships,is_authors_truncated,biblio,is_retracted,is_xpac,ids,"
               "locations,locations_count,cited_by_count,primary_location,abstract_inverted_index,created_date")
BATCH = 50


class QueryTimeout(RuntimeError):
    """The API stopped a query as too broad; a narrower one (fewer ids) succeeds."""


def wid(x):
    """'https://openalex.org/W123' | 'W123' | 123 -> 123"""
    if isinstance(x, int):
        return x
    return int(str(x).rstrip("/").rsplit("/", 1)[-1].lstrip("Ww"))


class OpenAlex:
    SEMANTIC_GAP = 1.05   # the API allows one semantic search per second per key

    def __init__(self, api_key=None, retries=8, timeout=60):
        self.key = api_key or os.environ.get("OPENALEX_API_KEY")
        self.retries, self.timeout = retries, timeout
        self.calls, self.cost_usd = 0, 0.0
        self._last_semantic = 0.0

    def get(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        if self.key:
            params["api_key"] = self.key
        query = urllib.parse.urlencode(params, safe=':,|*"')
        url = f"{API}/{path.lstrip('/')}?{query}"
        for attempt in range(self.retries):
            if "search.semantic" in params:
                time.sleep(max(0.0, self._last_semantic + self.SEMANTIC_GAP - time.time()))
                self._last_semantic = time.time()
            wait = 2 ** attempt
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "openalex-work-matcher"}),
                                            timeout=self.timeout) as r:
                    d = json.loads(r.read())
                self.calls += 1
                self.cost_usd += float((d.get("meta") or {}).get("cost_usd") or 0)
                return d
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {"results": [], "meta": {"count": 0}}
                body = e.read()
                if e.code == 504 and b"query_timeout" in body:
                    raise QueryTimeout(f"OpenAlex API query timeout for {path}")
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.retries - 1:
                    raise RuntimeError(f"OpenAlex API {e.code} for {path}: {body[:300]!r}")
                if e.code == 429:
                    try:
                        wait = max(float(json.loads(body).get("retryAfter") or 1), 1.0) * (1 + attempt)
                    except ValueError:
                        pass
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
            time.sleep(wait)

    def list(self, path, per_page=50, **params):
        d = self.get(path, per_page=per_page, **params)
        return d.get("results") or [], (d.get("meta") or {}).get("count") or 0

    def works(self, filter=None, search=None, semantic=None, per_page=50, select=WORK_FIELDS):
        kw = {"filter": filter, "search": search, "search.semantic": semantic}
        return self.list("works", per_page=per_page, select=select, include_xpac="true", **kw)

    def works_by_id(self, ids):
        """{id: API work} for the given work ids, in batches."""
        ids, out = sorted({wid(i) for i in ids}), {}
        for i in range(0, len(ids), BATCH):
            chunk = ids[i:i + BATCH]
            rs, _ = self.works(filter="openalex:" + "|".join(f"W{x}" for x in chunk), per_page=BATCH)
            out.update({wid(r["id"]): r for r in rs})
        return out

    def locations(self, filter, per_page=200):
        out, cursor = [], "*"
        while cursor:
            d = self.get("locations", filter=filter, per_page=per_page, cursor=cursor)
            out += d.get("results") or []
            cursor = (d.get("meta") or {}).get("next_cursor")
        return out

    def locations_of(self, ids):
        """{work id: [API location]} for the given work ids; a batch the API finds too broad is split in half."""
        out = {}

        def fetch(chunk):
            try:
                rows = self.locations("work_id:" + "|".join(f"W{x}" for x in chunk))
            except QueryTimeout:
                if len(chunk) == 1:
                    raise
                fetch(chunk[:len(chunk) // 2]); fetch(chunk[len(chunk) // 2:])
                return
            for l in rows:
                out.setdefault(wid(l["work_id"]), []).append(l)
        ids = sorted({wid(i) for i in ids})
        for i in range(0, len(ids), BATCH):
            fetch(ids[i:i + BATCH])
        return out


def abstract_text(inv):
    """Rebuild an abstract from its inverted index."""
    if not inv:
        return None
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def _bare_doi(d):
    if d is None:
        return None
    d = d.strip()
    for p in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"):
        if d.lower().startswith(p):
            return d[len(p):]
    return d


def same_location(l, record):
    return record is not None and l.get("provenance") == record.get("provenance") and l.get("native_id") == record.get("native_id")


def work_from_api(aw, locs, exclude=None):
    """API work + its /locations rows -> the matcher's work dict, shaped like the snapshot's work row the production SQL reads.
    The work's own `locations` list (with its version labels) comes from /works; each location's provenance, native id, ids
    and title key from its /locations row (same id). The title and publisher are the primary location's, as stored.
    `exclude`: a record whose own location is left out (a record already in OpenAlex, matched as if it were arriving)."""
    rows = {l.get("id"): l for l in (locs or [])}
    wl = aw.get("locations") or []
    kept = [l for l in wl if not same_location(rows.get(l.get("id")) or {}, exclude)]
    pl = aw.get("primary_location") or {}
    src = pl.get("source") or {}
    prow = rows.get(pl.get("id")) or {}
    ids = aw.get("ids") or {}
    auths = aw.get("authorships")

    def loc(l):
        r = rows.get(l.get("id")) or {}
        return {"provenance": r.get("provenance"), "native_id": r.get("native_id"), "doi": r.get("doi", l.get("doi")),
                "landing_page_url": l.get("landing_page_url"), "pdf_url": l.get("pdf_url"), "version": l.get("version"),
                "source_id": (l.get("source") or {}).get("id"), "ids": r.get("ids") or [],
                "title_author": (r.get("merge_key") or {}).get("title_author")}
    return {
        "id": wid(aw["id"]), "doi": aw.get("doi"), "title": prow.get("title") or aw.get("title"),
        "publication_year": aw.get("publication_year"), "type": aw.get("type"), "language": aw.get("language"),
        "publisher": prow.get("publisher"), "biblio": aw.get("biblio") or {}, "is_retracted": aw.get("is_retracted"),
        "is_xpac": aw.get("is_xpac"), "abstract": abstract_text(aw.get("abstract_inverted_index")),
        "cited_by_count": aw.get("cited_by_count"),
        "locations_count": None if aw.get("locations_count") is None else aw["locations_count"] - (len(wl) - len(kept)),
        "created_date": aw.get("created_date"),
        "ids": {k: (_bare_doi(v) if k == "doi" else v) for k, v in ids.items()},
        "authorships": None if auths is None else [{"display_name": (a.get("author") or {}).get("display_name"),
                                                    "orcid": (a.get("author") or {}).get("orcid")} for a in auths],
        "n_authorships": None if auths is None else len(auths),
        "locations": [loc(l) for l in kept],
        "primary_location": {"provenance": prow.get("provenance"), "version": pl.get("version"), "source_name": src.get("display_name"),
                             "raw_source_name": pl.get("raw_source_name"), "landing_page_url": pl.get("landing_page_url")},
    }

"""Candidate works for one record, through the public API: the production channels (`pipeline/sql/candidates.sql`) with the
corpus-wide indexes replaced by API queries.

  id    shares a DOI, PMID, URL or repository native id (key on <= 50 works)   top 20 by title Jaccard
  ti    identical normalized title (title on <= 5000 works)                    top 20 by same first author, then year gap
  ay    same first-author key, year within 2                                  top 5 by title Jaccard (> 0)
  mh    a shared MinHash band, Jaccard >= 0.4, year gap <= 3                   top 5 by title Jaccard
  emb   nearest by embedding                                                  top 10 by cosine
  bound the title rule's pick (the work already holding the record's title + author key)

Production looks titles, authors and bands up in its own index of every work; here title search, author search and semantic
search return a pool and the same rules choose from it, so a channel can miss a work production would find (the README's
known differences). `as_of` drops works created on or after that date, so a record already in OpenAlex can be matched as of
the night it arrived.
"""
import re

from . import embed as E, minhash, openalex as OA, text, titlekey

POOL = 50
EMB_POOL = 50


def _q(s):
    """A value safe inside an API filter or search: no filter separators (commas, pipes), no search syntax (wildcards, quotes,
    brackets, Boolean operators)."""
    s = re.sub(r"[,|*?\"()\[\]{}:!\\^~]+", " ", s or "")
    s = re.sub(r"\b(AND|OR|NOT)\b", lambda m: m.group(0).lower(), s)
    return re.sub(r"\s+", " ", s).strip()


def _fit(s, max_bytes=6000):
    """Shorten a query so its URL-encoded form stays under the API's 8 KB URL limit."""
    import urllib.parse
    while s and len(urllib.parse.quote(s)) > max_bytes:
        s = s[:int(len(s) * 0.8)]
    return s


def _tok(nt):
    seen = []
    for z in (nt or "").split(" "):
        if len(z) >= 2 and z not in seen:
            seen.append(z)
    return seen


def jaccard(a, b):
    if a is None or b is None:
        return None
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


class Finder:
    def __init__(self, api=None, embedder=None):
        self.api = api or OA.OpenAlex()
        self.embedder = embedder

    def _keep(self, w, as_of):
        return as_of is None or (w.get("created_date") or "")[:10] < as_of

    def find(self, rec, as_of=None, exclude_self=False, bad_titles=frozenset()):
        """-> (candidates [{cid, work, channels, emb_rank, emb_score}], context {title_pick, nt_fanout, calls})"""
        api = self.api
        nt = text.norm_title(rec.get("title"))
        akey = text.author_key(text.record_a0(rec.get("authors")))
        keys = text.record_keys(rec)
        yr = _year(rec)
        title = text.TAG.sub(" ", rec.get("title") or "").strip()
        pool = {}
        # OR over the title words (production's channels need only some shared words; the API's default search needs all)
        orq = _fit(" OR ".join(_q(t) for t in text.title_tokens(nt)[:30] if _q(t)), 3000)

        def add(rs):
            for r in rs:
                if self._keep(r, as_of):
                    pool.setdefault(OA.wid(r["id"]), r)

        # id: the works holding each of the record's keys
        idc = set()
        for k in keys:
            got = set()
            if k.startswith("d:"):
                got = {OA.wid(l["work_id"]) for l in api.locations(f"doi:{_q(k[2:])}") if l.get("work_id")
                       and not (exclude_self and OA.same_location(l, rec))}
            elif k.startswith("p:"):
                rs, _ = api.works(filter=f"pmid:{k[2:]}"); add(rs); got = {OA.wid(r["id"]) for r in rs}
            elif k.startswith("n:") and rec.get("native_id"):
                got = {OA.wid(l["work_id"]) for l in api.locations(f"native_id:{_q(rec['native_id'])}") if l.get("work_id")
                       and not (exclude_self and OA.same_location(l, rec))}
            elif k.startswith("u:"):
                for url in (rec.get("landing_page_url"),):
                    if url and text.url_key(url) == k:
                        rs, _ = api.works(filter=f"locations.landing_page_url:{_q(url)}"); add(rs); got |= {OA.wid(r["id"]) for r in rs}
            if 0 < len(got) <= 50:
                idc |= got
        # title, author and semantic pools
        s1 = s2 = []
        n_title_hits = 0
        if title:
            s1, n_title_hits = api.works(filter=f"title.search:{_fit(_q(title), 3000)}", per_page=POOL); add(s1)
            if orq:
                s2, _ = api.works(search=orq, per_page=POOL); add(s2)
        ay_pool = []
        if akey and yr is not None:
            ay_pool, _ = api.works(filter=f"raw_author_name.search:{_q(akey.split(' ', 1)[1])},publication_year:{yr - 2}-{yr + 2}",
                                   search=orq or None, per_page=POOL); add(ay_pool)
        emb_pool = []
        rtext = E.record_text(rec)
        if rtext and self.embedder is not None:
            emb_pool, _ = api.works(semantic=_fit(_q(rtext)), per_page=EMB_POOL); add(emb_pool)
        ids_needed = set(pool) | idc
        missing = ids_needed - set(pool)
        if missing:
            add(api.works_by_id(missing).values())
        locs = api.locations_of(list(pool))
        excl = rec if exclude_self else None
        works = {i: OA.work_from_api(pool[i], locs.get(i, []), exclude=excl) for i in pool}
        side = {i: (text.norm_title(w.get("title")), _first_akey(w), w.get("publication_year")) for i, w in works.items()}
        rtok = _tok(nt)
        jac = {i: jaccard(rtok, _tok(s[0])) if nt and s[0] else None for i, s in side.items()}
        gap = {i: (abs(s[2] - yr) if s[2] is not None and yr is not None else None) for i, s in side.items()}
        rank = lambda i: (-(jac[i] if jac[i] is not None else -1), gap[i] if gap[i] is not None else 99, i)
        ch = {}

        def put(i, c):
            ch.setdefault(i, set()).add(c)
        for i in sorted((i for i in idc if i in works), key=rank)[:20]:
            put(i, "id")
        exact = [i for i in works if nt and side[i][0] == nt]
        n_exact_title = len([r for r in s1 if text.norm_title(r.get("title")) == nt]) if nt else 0
        nt_fanout = (n_title_hits if s1 and n_exact_title == len(s1) and n_title_hits > len(s1) else (len(exact) or None))
        if nt_fanout is None or nt_fanout <= 5000:
            for i in sorted(exact, key=lambda i: (-(int(side[i][1] == akey) if side[i][1] and akey else 0),
                                                  gap[i] if gap[i] is not None else 99, i))[:20]:
                put(i, "ti")
        if akey and yr is not None:
            ay = [i for i in works if side[i][1] == akey and side[i][2] is not None and abs(side[i][2] - yr) <= 2 and (jac[i] or 0) > 0]
            for i in sorted(ay, key=rank)[:5]:
                put(i, "ay")
        rb = minhash.bands(nt)
        if rb:
            rbs = set(rb)
            mh = [i for i in works if (jac[i] or 0) >= 0.4 and (gap[i] or 0) <= 3 and set(minhash.bands(side[i][0]) or ()) & rbs]
            for i in sorted(mh, key=rank)[:5]:
                put(i, "mh")
        # title rule's pick: the work(s) whose locations carry the record's title + author key
        tkey = titlekey.title_author(rec)
        title_pick = None
        if tkey and len(tkey) > 20 and (rec.get("title") or "").strip() not in bad_titles:
            holders = sorted(i for i, w in works.items() if any(l.get("title_author") == tkey for l in w["locations"]))
            if 0 < len(holders) <= 3:
                conflict = rec.get("provenance") in ("crossref", "datacite") and any(
                    i.get("namespace") == "doi" for i in rec.get("ids") or []) and any(
                    l.get("provenance") == rec["provenance"] for l in works[holders[0]]["locations"])
                if not conflict:
                    title_pick = holders[0]
                    put(title_pick, "bound")
        # emb: the semantic pool scored with the production embedding recipe, top 10 by cosine
        emb = {}
        if emb_pool:
            cand = [OA.wid(r["id"]) for r in emb_pool if OA.wid(r["id"]) in works]
            vecs = self.embedder([rtext] + [E.work_text(works[i]) for i in cand])
            scored = sorted(((E.cos(vecs[0], v), i) for i, v in zip(cand, vecs[1:])), key=lambda t: (-t[0], t[1]))[:10]
            for r, (c, i) in enumerate(scored, 1):
                emb[i] = (r, (1 + c) / 2)
                put(i, "emb")
        cands = [{"cid": i, "work": works[i], "channels": sorted(c - {"emb"}) + (["emb"] if "emb" in c else []),
                  "emb_rank": emb.get(i, (None, None))[0], "emb_score": emb.get(i, (None, None))[1]} for i, c in sorted(ch.items())]
        return cands, {"title_pick": title_pick, "nt_fanout": nt_fanout, "title_key": tkey}


def _year(rec):
    from .features import year
    return year(rec.get("published_date", rec.get("year")))


def _first_akey(w):
    a = (w.get("authorships") or [None])[0]
    return text.author_key((a or {}).get("display_name")) if a else None

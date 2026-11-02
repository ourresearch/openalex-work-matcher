"""Text keys, exactly as the production SQL builds them (Spark SQL semantics: `translate`, `lower`, Java regex classes).

nt    normalized title: markup stripped, common accents folded, lower case, letters and digits only, single spaces
akey  first-author key "initial surname"
keys  identity keys: d: DOI, p: PMID, u: URL, n: repository native id
"""
import re

ACCENTS = str.maketrans("ÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕØÚÙÛÜÝáàâäãåçéèêëíìîïñóòôöõøúùûüýÿ",
                        "AAAAAACEEEEIIIINOOOOOOUUUUYaaaaaaceeeeiiiinoooooouuuuyy")
STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "its", "are", "was", "were", "that", "this", "des", "les", "und", "der", "die",
    "das", "von", "del", "los", "las", "une", "por", "para", "con", "dans", "sur", "aux", "ein", "eine", "den", "het", "een",
    "van", "som", "och"}
TAG = re.compile(r"<[^>]+>")
DOI_URL = re.compile(r"^https?://(dx\.)?doi\.org/")
URL_HEAD = re.compile(r"^https?://(www\.)?")
TRAILING_SLASH = re.compile(r"/+$")
ORCID = re.compile(r"([0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9Xx])")
DIGITS_END = re.compile(r"(\d+)$")


def _ln(c):
    """Java \\p{L} or \\p{N}."""
    return c.isalpha() or c.isnumeric()


def ln_tokens(s):
    """split(regexp_replace(s, '[^\\p{L}\\p{N}]+', ' '), ' ') without the empty strings."""
    out, cur = [], []
    for c in s:
        if _ln(c):
            cur.append(c)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


def letters(s):
    """regexp_replace(s, '[^\\p{L}]', '')"""
    return "".join(c for c in s if c.isalpha())


def strim(s):
    """Spark trim: ASCII spaces only."""
    return s.strip(" ")


def nullif_trim(s):
    """nullif(trim(s), '')"""
    if s is None:
        return None
    s = strim(str(s))
    return s or None


def norm_title(title):
    """nt; None when the title normalizes to nothing."""
    s = TAG.sub(" ", title or "").translate(ACCENTS).lower()
    return " ".join(ln_tokens(s)) or None


def _surname_initial(a0):
    if a0 is None:
        return None, None
    if "," in a0:
        sur_src, ini_src = a0.split(",")[0], (a0.split(",") + [""])[1]
    else:
        parts = re.split(" +", strim(a0))
        sur_src, ini_src = parts[-1], parts[0]
    return letters(sur_src), letters(ini_src)[:1]


def author_key(a0):
    """akey from the first author's name as the SQL reads it (a0, before folding)."""
    if a0 is None:
        return None
    sur, ini = _surname_initial(a0.translate(ACCENTS).lower())
    return f"{ini} {sur}" if len(sur) >= 2 and len(ini) == 1 else None


def record_a0(authors):
    """coalesce(nullif(concat_ws(', ', family, given), ''), name) of the record's first author."""
    if not authors:
        return None
    a = authors[0] or {}
    fg = ", ".join(x for x in (a.get("family"), a.get("given")) if x is not None)
    return fg or a.get("name")


def url_key(url):
    if url is None:
        return None
    u = url.lower()
    if DOI_URL.search(u):
        return "d:" + DOI_URL.sub("", u)
    return "u:" + TRAILING_SLASH.sub("", URL_HEAD.sub("", u))


def pdf_key(url):
    return None if url is None else "u:" + TRAILING_SLASH.sub("", URL_HEAD.sub("", url.lower()))


def _clean_keys(ks):
    out = []
    for k in ks:
        if k is not None and k not in ("d:", "p:", "u:", "n:") and k not in out:
            out.append(k)
    return out


def record_keys(rec):
    """Identity keys of an incoming record (its own DOI, PMIDs, landing page, PDF, repository native id)."""
    ids = rec.get("ids") or []
    ks = ["d:" + i["id"].lower() for i in ids if i.get("namespace") == "doi" and i.get("relationship") == "self" and i.get("id") is not None]
    ks += ["p:" + (m.group(1) if (m := DIGITS_END.search(i["id"])) else "") for i in ids
           if i.get("namespace") == "pmid" and i.get("id") is not None]
    ks += [url_key(rec.get("landing_page_url")), pdf_key(rec.get("pdf_url")),
           "n:" + rec["native_id"].lower() if rec.get("provenance") not in ("crossref", "datacite", None) and rec.get("native_id") is not None else None]
    return _clean_keys(ks)


def work_keys(work):
    """Identity keys of a work: its DOI and PMID, and every location's landing page, PDF and repository native id."""
    ids = work.get("ids") or {}
    pmid = ids.get("pmid")
    ks = ["d:" + ids["doi"].lower() if ids.get("doi") is not None else None,
          "p:" + (m.group(1) if (m := DIGITS_END.search(pmid)) else "") if pmid is not None else None]
    locs = work.get("locations") or []
    ks += [url_key(l.get("landing_page_url")) for l in locs]
    ks += [pdf_key(l.get("pdf_url")) for l in locs]
    ks += ["n:" + l["native_id"].lower() if l.get("provenance") not in ("crossref", "datacite", None) and l.get("native_id") is not None else None
           for l in locs]
    return _clean_keys(ks)


def title_tokens(nt):
    """Title tokens >= 3 characters minus stopwords, in first-seen order (the MinHash input)."""
    seen = []
    for x in (nt or "").split(" "):
        if len(x) >= 3 and x not in STOPWORDS and x not in seen:
            seen.append(x)
    return seen


def surname(name, family=None):
    """The `su` token of one author: last word of the family name (else of the name before a comma, else of the name),
    accents folded, lower case, letters only."""
    if nullif_trim(family) is not None:
        src = family
    elif name is not None and "," in name:
        src = name.split(",")[0]
    else:
        src = name
    if src is None:
        return None
    return letters(re.split(" +", strim(src))[-1].translate(ACCENTS).lower()) or None


def orcid(s):
    if s is None:
        return None
    m = ORCID.search(s)
    return m.group(1).upper() if m else None


def bare_doi(s, trim=True):
    """regexp_replace(lower(trim(s)), '^https?://(dx\\.)?doi\\.org/', '')"""
    if s is None:
        return None
    return DOI_URL.sub("", (strim(s) if trim else s).lower())

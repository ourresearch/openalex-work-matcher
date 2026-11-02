"""The title + first-author key (`merge_key.title_author` on every OpenAlex location), as the ingest pipeline builds it.

The old title rule bound a record to the work already holding its key. The matcher no longer decides on it, but the work
holding the record's key (the "title rule's pick") is one of XGBoost's inputs (`is_bound`) and is always a candidate.
"""
import re
import unicodedata
from functools import lru_cache

from nameparser import HumanName   # nameparser==2.1.0, as in the ingest pipeline


def _remove_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_title(title):
    if not title:
        return ""
    t = title[0:500].lower()
    t = _remove_accents(t)
    t = re.sub(r"<\w+.*?>", "", t)
    t = re.sub(r"\b(the|a|an|of|to|in|for|on|by|with|at|from|\n)\b", "", t)
    return "".join(e for e in t if e.isalpha() or e.isdigit()).strip()


def _remove_latin_characters(author):
    if author and any("\u0080" <= c <= "ʯ" for c in author):
        author = unicodedata.normalize("NFKD", author).encode("ascii", "ignore").decode("ascii")
    return author


def _remove_author_prefixes(author):
    if not author:
        return ""
    for prefix in ("None ", "Array "):
        if author.startswith(prefix):
            author = author.replace(prefix, "")
    return author


def _clean(name):
    if not name:
        return ""
    return re.sub(r"[ \-‐.'' ́>]", "", name).strip()


@lru_cache(maxsize=100_000)
def author_key(raw_name):
    """'surname;i' from one author's name string."""
    if not raw_name:
        return None
    a = _remove_author_prefixes(_remove_latin_characters(raw_name))
    h = HumanName(a)
    first, last = _clean(h.first), _clean(h.last)
    k = f"{last};{first[0] if first else ''}".lower()
    return k or None


def title_author(rec):
    """The record's key, or None when its title normalizes to nothing."""
    nt = normalize_title(rec.get("title"))
    if not nt:
        return None
    authors = rec.get("authors") or []
    a = authors[0] if authors else None
    if not a or "".join(x or "" for x in (a.get("given"), a.get("family"), a.get("name"))) == "":
        return nt
    raw = a.get("name") or f"{a.get('given') or ''} {a.get('family') or ''}".strip()
    k = author_key(raw)
    return f"{nt}_{k}" if k else nt

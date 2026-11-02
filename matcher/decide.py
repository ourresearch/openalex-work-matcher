"""The decision for one record, from its XGBoost and Qwen scores (`pipeline/sql/decisions.sql` and the copy sums step).

no candidate, or XGBoost's best < 0.01   -> new work (layers no_candidates, xgb_mint)
Qwen's best >= 0.910                      -> that work (qwen_bind)
Qwen's best plus the candidates that are copies of it >= 0.910 -> that work (qwen_copy_bind)
the record alone fills Qwen's token cap   -> new work (qwen_over_cap)
otherwise                                 -> new work (qwen_mint)

Copy sums: Qwen was trained to say yes to at most one candidate, so when a shortlist holds two or three copies of one paper it
splits its yes between them and none reaches 0.910 on its own.
"""
import difflib
import re
import unicodedata

from .model import MINT_BELOW

BIND_AT = 0.910


def _ascii(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def is_copy(a, b):
    """Two candidate works are copies of one paper: folded titles equal (or difflib ratio >= 0.93) and >= 20 folded characters,
    a shared author name word when both list authors (equal titles when either has none), years within 1 when both have one."""
    ta, tb = re.sub(r"[^a-z0-9]", "", _ascii(a.get("title"))), re.sub(r"[^a-z0-9]", "", _ascii(b.get("title")))
    if min(len(ta), len(tb)) < 20 or (ta != tb and difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio() < 0.93):
        return False
    wa = {w for x in a.get("authors") or [] for w in re.findall(r"[a-z]{3,}", _ascii(x))}
    wb = {w for x in b.get("authors") or [] for w in re.findall(r"[a-z]{3,}", _ascii(x))}
    if (wa and wb and not wa & wb) or ((not wa or not wb) and ta != tb):
        return False
    ya, yb = a.get("year"), b.get("year")
    return not ya or not yb or abs(ya - yb) <= 1


def copy_sum(cands):
    """cands [(cid, p, summary)] -> (top cid, its p plus the p of the candidates that are copies of it)."""
    cid, p, top = max(cands, key=lambda c: (c[1], c[0]))
    return cid, p + sum(pp for c, pp, s in cands if c != cid and is_copy(top, s))


def decide(xgb, qwen=None, over_cap=False, summaries=None):
    """xgb: [(cid, p)] for every scored pair; qwen: [(cid, p)] for the shortlist (None if Qwen was not run);
    summaries: {cid: {"title", "authors", "year"}} for the copy check. Returns the decision dict."""
    out = {"work": None, "xgb_work": None, "xgb_p": None, "qwen_work": None, "qwen_p": None}
    if not xgb:
        return {**out, "layer": "no_candidates"}
    xc, xp = max(xgb, key=lambda t: (t[1], -t[0]))
    out.update(xgb_work=xc, xgb_p=xp)
    if xp < MINT_BELOW:
        return {**out, "layer": "xgb_mint"}
    scored = [(c, p) for c, p in (qwen or []) if p is not None]
    if not scored:
        if over_cap:
            return {**out, "layer": "qwen_over_cap"}
        raise ValueError("a record at >= 0.01 needs its Qwen scores")
    qc, qp = max(scored, key=lambda t: (t[1], t[0]))
    out.update(qwen_work=qc, qwen_p=qp)
    if qp >= BIND_AT:
        return {**out, "layer": "qwen_bind", "work": qc}
    if sum(p for _, p in scored) >= BIND_AT and summaries is not None:
        c, s = copy_sum([(c, p, summaries[c]) for c, p in scored])
        if s >= BIND_AT:
            return {**out, "layer": "qwen_copy_bind", "work": c, "copy_sum": s}
    return {**out, "layer": "qwen_mint"}

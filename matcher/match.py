"""One record in, one decision out: identifiers, then candidates, features, hard rules, XGBoost, Qwen and the decision rule.

    from matcher.match import Matcher
    m = Matcher(adapter="assets/qwen-adapter")          # loads the embedding model, XGBoost and Qwen
    m.match(record)  ->  {"work": 2146292423 | None, "layer": "qwen_bind", "xgb_p": .., "qwen_p": .., "shortlist": [..]}

`work` None means OpenAlex would make the record a new work. Stage 1 (everything but Qwen) runs on a CPU; `stage1()` returns
the Qwen prompt so the GPU part can run elsewhere and `finish()` completes the decision.
"""
import os

from . import candidates as C, decide as D, embed as E, features as F, identifiers as I, model as M, openalex as OA, prompt as P

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_bad_titles(path=os.path.join(HERE, "models", "bad_titles.txt")):
    if not os.path.exists(path):
        return frozenset()
    with open(path) as f:
        return frozenset(l.rstrip("\n").strip() for l in f if l.strip())


class Matcher:
    def __init__(self, adapter=None, api=None, embedder=None, device=None, load_qwen=True):
        self.api = api or OA.OpenAlex()
        self.embedder = embedder or E.Embedder(device=device)
        self.finder = C.Finder(self.api, self.embedder)
        self.ids = I.Identifiers(self.api)
        self.xgb, self.veto, self.bad = M.XGBoostScorer(), M.load_veto(), load_bad_titles()
        self.qwen = None
        if load_qwen and adapter:
            from .qwen import QwenScorer
            self.qwen = QwenScorer(adapter, device=device or "cuda")

    def stage1(self, rec, as_of=None, exclude_self=False, identifiers=True):
        """Everything up to the Qwen prompt. as_of / exclude_self: match a record already in OpenAlex as of the night it
        arrived (drop works created since and the record's own location)."""
        if identifiers:
            w, src = self.ids.match(rec, as_of=as_of, exclude_self=exclude_self)
            if w is not None:
                return {"done": True, "decision": {"work": w, "layer": f"identifier:{src}"}}
        cands, ctx = self.finder.find(rec, as_of=as_of, exclude_self=exclude_self, bad_titles=self.bad)
        rows = F.features(rec, cands, bound_work=ctx["title_pick"], nt_fanout=ctx["nt_fanout"])
        rdo = F.record_side(rec)["tok"].get("do", set())
        by = {c["cid"]: c for c in cands}
        kept = [r for r in rows if not M.vetoed(r, rdo, F.work_side(by[r["cid"]]["work"])["tok"].get("do", set()), self.veto)]
        ps = self.xgb.score(kept)
        xgb = sorted(zip([r["cid"] for r in kept], ps), key=lambda t: (-t[1], t[0]))
        out = {"done": False, "xgb": xgb, "candidates": {c["cid"]: c for c in cands}, "context": ctx,
               "n_candidates": len(cands), "n_vetoed": len(rows) - len(kept)}
        if not xgb or xgb[0][1] < M.MINT_BELOW:
            out["done"] = True
            out["decision"] = D.decide(xgb)
            return out
        short = [(c, by[c]["work"]) for c, _ in xgb[:P.MAX_CANDIDATES]]
        out["prompt"] = P.build(rec, short)
        out["summaries"] = {c: {k: P.work_summary(w)[k] for k in ("title", "authors", "year")} for c, w in short}
        return out

    def finish(self, s1, qwen_scores, over_cap=False):
        """Complete a stage-1 result with its Qwen scores [(cid, p)]."""
        if s1["done"]:
            return s1["decision"]
        d = D.decide(s1["xgb"], [(int(k), p) for k, p in qwen_scores], over_cap=over_cap, summaries=s1["summaries"])
        d["shortlist"] = [{"work": c, "xgb_p": p} for c, p in s1["xgb"][:P.MAX_CANDIDATES]]
        return d

    def match(self, rec, **kw):
        s1 = self.stage1(rec, **kw)
        if s1["done"]:
            return s1["decision"]
        if self.qwen is None:
            raise RuntimeError("this record needs the Qwen stage: pass adapter= (and run on a GPU)")
        scores, over = self.qwen.score(s1["prompt"])
        return self.finish(s1, scores, over_cap=over)

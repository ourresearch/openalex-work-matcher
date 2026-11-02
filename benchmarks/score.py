"""Recompute the matcher's benchmark numbers from the files in benchmarks/data/. Standard library only; run from the repo root.

  python3 benchmarks/score.py nights     the shipping matcher on 8 judged nights of real incoming records
  python3 benchmarks/score.py baseline   the title + author rule it replaced, on the same nights
  python3 benchmarks/score.py gold       the frozen gold set (pairwise labels, human-reviewed disputes)
  python3 benchmarks/score.py heldout    the held-out records the bind threshold was set on
  python3 benchmarks/score.py all

The matcher's decision for a record, as production makes it (pipeline/sql/decisions.sql):
  no candidate, or XGBoost's best candidate < 0.01         -> new work
  Qwen's best candidate >= 0.910                           -> bind it
  Qwen's best p + the p of the candidates that are copies
  of it (same paper, see matcher/decide.py) >= 0.910        -> bind it
  otherwise                                                -> new work
Precision = weighted share of binds that join the right work. Found (recall) = weighted share of records that have a matching
work among their candidates that the matcher binds to the right work. Weights undo the sampling (benchmarks/README.md)."""
import collections
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TAU, MINT_BELOW = 0.910, 0.01
JOIN = ("same", "version_preprint")


def rows(path):
    with gzip.open(os.path.join(DATA, path), "rt") as f:
        for line in f:
            yield json.loads(line)


def wsum(xs):
    return sum(r["w"] for r in xs)


def share(xs, ok):
    total = wsum(xs)
    return sum(r["w"] for r in xs if ok(r)) / total if total else float("nan")


def decide(qs, copies, xgb_top, over_cap=False, copy_sums=True):
    """qs: [(work, p)] in shortlist order; copies: {work: [works it is a copy of]}. -> (bound work or None, layer)"""
    if xgb_top is None:
        return None, "no_candidates"
    if xgb_top < MINT_BELOW:
        return None, "xgb_mint"
    if over_cap:
        return None, "qwen_over_cap"
    if not qs:
        return "MISSING", "qwen_missing"
    work, p = max(qs, key=lambda wp: wp[1])
    if p >= TAU:
        return work, "qwen_bind"
    if copy_sums:
        s = p + sum(pp for w, pp in qs if w != work and w in copies.get(work, ()))
        if s >= TAU:
            return work, "qwen_copy_bind"
    return None, "qwen_mint"


def pct(x):
    return f"{100 * x:5.1f}%"


# ------------------------------------------------------------------------------------------------ judged nights
EIGHT = ["2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-26", "2026-10-27", "2026-10-28"]
SEVEN = ["2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]


def night_votes(fn):
    """Last valid verdict per shortlist key -> (picked work or None, works shown)."""
    v = {}
    for r in rows(f"nights/judges/{fn}"):
        if r["ok"]:
            a = r["answer"]
            pick = r["shortlist"].get(a.get("pick")) if a["label"] in JOIN else None
            v[(r["night"], r["record_key"], r["shortlist_hash"], r["prompt"])] = (pick, set(r["shortlist"].values()))
    return v


def night_records():
    """Judged records with their label: the panel (Opus and a second judge agree, a third breaks a tie) where the second
    judge voted, Opus alone elsewhere; three-way splits are left out."""
    opus, sol, fable = night_votes("opus.jsonl.gz"), night_votes("sol.jsonl.gz"), night_votes("fable.jsonl.gz")
    key_of = {(k[0], k[1]): k for k in opus}
    out = []
    for s in rows("nights/records.jsonl.gz"):
        k = key_of.get((s["night"], s["record_key"]))
        if k is None:
            continue
        o, seen = opus[k]
        if k not in sol or sol[k][0] == o:
            label = o
        elif k in fable and fable[k][0] in (o, sol[k][0]):
            label = fable[k][0]
        else:
            continue
        out.append({**s, "k": (s["night"], s["record_key"]), "label": label, "seen": seen, "opus": o,
                    "w": s["weight"] * s["cell_n"] / s["cell_drawn"],
                    "cellkey": (s["night"], s["prov"], s["title_rule"], s["cell"])})
    return out


def night_inputs():
    top = collections.defaultdict(float)
    for r in rows("nights/xgboost.jsonl.gz"):
        k = (r["night"], r["record_key"])
        top[k] = max(top[k], r["p"])
    qs, copies = collections.defaultdict(list), collections.defaultdict(dict)
    for r in rows("nights/qwen.jsonl.gz"):
        k = (r["night"], r["record_key"])
        if r["p"] is not None:
            qs[k].append((r["work"], r["p"]))
        copies[k][r["work"]] = set(r["copy_of"])
    with open(os.path.join(DATA, "nights", "over_cap.json")) as f:
        over = {(o["night"], o["record_key"]) for o in json.load(f)}
    pair = collections.defaultdict(dict)
    for r in rows("nights/judges/opus_pairwise.jsonl.gz"):
        if r["ok"]:
            pair[((r["night"], r["record_key"]), r["work"])][r["order"]] = r["answer"]["label"]
    same = lambda k, w: len(pair.get((k, w), {})) == 2 and all(x in JOIN for x in pair[(k, w)].values())
    return top, qs, copies, over, same


def boot(recs, metric, n=400):
    cells = collections.defaultdict(list)
    for r in recs:
        cells[r["cellkey"]].append(r)
    rnd = random.Random(1668)
    vals = sorted(metric([rnd.choice(c) for c in cells.values() for _ in c]) for _ in range(n))
    return vals[int(0.05 * n)], vals[int(0.95 * n) - 1]


def nights():
    recs = night_records()
    top, qs, copies, over, same = night_inputs()
    print("== The matcher as it ships, on 8 judged nights of incoming records")
    print("   (records arriving 18-22 and 26-28 October 2026; label = the judges' pick, a bind to a different work is checked")
    print("   pair by pair by two blind judgments, so a bind to another copy of the same paper counts as right)")
    for copy_sums in (False, True):
        rs = []
        for r in recs:
            if r["night"] not in EIGHT:
                continue
            k = r["k"]
            m, layer = decide(qs.get(k, []), copies.get(k, {}), top.get(k), k in over, copy_sums)
            if m != "MISSING":
                rs.append({**r, "m": m, "layer": layer})
        right = lambda r: r["m"] == r["label"] or same(r["k"], r["m"])
        binds, matched = [r for r in rs if r["m"]], [r for r in rs if r["label"]]
        prec = lambda xs: share([r for r in xs if r["m"]], right)
        found = lambda xs: share([r for r in xs if r["label"]], lambda r: bool(r["m"]) and right(r))
        name = "with copy sums (shipping)" if copy_sums else "without copy sums"
        lo, hi = boot(rs, prec)
        print(f"  {name:28s} precision {pct(prec(rs))} (90% range {pct(lo).strip()}-{pct(hi).strip()})  found {pct(found(rs))}  "
              f"judged records {len(rs):,}, binds {len(binds):,}, with a match {len(matched):,}")
        if copy_sums:
            print("  by source of the incoming record:")
            for p in ("crossref", "datacite", "pubmed", "repo", "other"):
                xs = [r for r in rs if r["prov"] == p]
                if xs:
                    print(f"    {p:10s} precision {pct(prec(xs))}  found {pct(found(xs))}  (judged {len(xs):,}, binds "
                          f"{sum(1 for r in xs if r['m']):,})")
            for g, ns in (("replay nights 18-22 Oct", EIGHT[:5]), ("live nights 26-28 Oct", EIGHT[5:])):
                xs = [r for r in rs if r["night"] in ns]
                print(f"    {g:24s} precision {pct(prec(xs))}  found {pct(found(xs))}")
            short = [r for r in rs if r["short_title"]]
            print(f"    titles of 3 words or fewer: precision {pct(prec(short))}  found {pct(found(short))}  (judged {len(short):,})")
    return recs


def baseline():
    recs = night_records()
    _, _, _, _, same = night_inputs()
    print("== The title + author rule the matcher replaced, on the same judged records")
    rs = [{**r, "m": r["title_rule_work"]} for r in recs if r["night"] in EIGHT]
    right = lambda r: r["m"] == r["label"] or same(r["k"], r["m"])
    print(f"  8 nights, same labels         precision {pct(share([r for r in rs if r['m']], right))}  "
          f"found {pct(share([r for r in rs if r['label']], lambda r: bool(r['m']) and right(r)))}  (binds {sum(1 for r in rs if r['m']):,})")
    # the first read-out: 7 replay nights, Opus alone, every judged record (16-22 October)
    pick = {}
    for r in rows("nights/judges/opus.jsonl.gz"):
        if r["ok"]:
            a = r["answer"]
            pick[(r["night"], r["record_key"])] = r["shortlist"].get(a.get("pick")) if a["label"] in JOIN else None
    rs = []
    for s in rows("nights/records.jsonl.gz"):
        k = (s["night"], s["record_key"])
        if s["night"] in SEVEN and k in pick:
            rs.append({**s, "w": s["weight"] * s["cell_n"] / s["cell_drawn"], "p": pick[k], "m": s["title_rule_work"]})
    print(f"  7 nights 16-22 Oct, Opus alone precision {pct(share([r for r in rs if r['m']], lambda r: r['m'] == r['p']))}  "
          f"found {pct(share([r for r in rs if r['p']], lambda r: r['m'] == r['p']))}  (judged {len(rs):,})")


# ------------------------------------------------------------------------------------------------ gold and held-out
def shortlists(path, y_field=None):
    qs, copies, ys = collections.defaultdict(list), collections.defaultdict(dict), collections.defaultdict(dict)
    for r in rows(path):
        k = r["record_key"]
        if r["p"] is not None:
            qs[k].append((r["work"], r["p"]))
        copies[k][r["work"]] = set(r["copy_of"])
        if y_field:
            ys[k][r["work"]] = r[y_field]
    return qs, copies, ys


def graded(recs, choose, label_of, has_match):
    """Weighted precision over binds whose pair has a label, found over records with a labelled match."""
    right = bound = pos = 0.0
    binds = wrong = 0
    for k, w in recs:
        pos += w * has_match(k)
        m = choose(k)
        if m is None:
            continue
        binds += 1
        y = label_of(k, m)
        if y is None:
            continue
        bound += w
        right += w * (y == 1)
        wrong += y == 0
    return right / bound if bound else float("nan"), right / pos if pos else float("nan"), binds, wrong


def gold():
    lab = {(r["record_key"], r["work"]): r["label"] for r in rows("gold/labels.jsonl.gz")}
    weight = {r["record_key"]: r["weight"] for r in rows("gold/records.jsonl.gz")}
    xs = {"p": collections.defaultdict(dict), "p_earlier": collections.defaultdict(dict)}
    for r in rows("gold/xgboost.jsonl.gz"):
        for col, d in xs.items():
            if r[col] is not None:
                d[r["record_key"]][r["work"]] = r[col]
    with open(os.path.join(DATA, "gold", "over_cap.json")) as f:
        over = {o["record_key"] for o in json.load(f)}
    has = lambda k: any(lab.get((k, w)) == 1 for w in xs["p"][k])
    print("== Frozen gold: 3,359 records drawn from three nights of incoming records, every candidate pair labelled")
    for title, col, qfile, cap in (("the shipping matcher (its XGBoost's 0.01 cut and top 10, then Qwen)", "p", "gold/qwen.jsonl.gz", over),
                                   ("for reference: the earlier XGBoost's cut and top 10, the same Qwen", "p_earlier",
                                    "gold/qwen_earlier.jsonl.gz", set())):
        xg = xs[col]
        qs, copies, _ = shortlists(qfile)
        recs = [(k, weight[k]) for k in xg if k in weight]
        print(f"  {title}")
        for copy_sums in (False, True):
            choose = lambda k: decide(qs.get(k, []), copies.get(k, {}), max(xg[k].values()), k in cap, copy_sums)[0]
            pick = {k: choose(k) for k, _ in recs}
            assert "MISSING" not in pick.values(), "a record past the 0.01 cut has no Qwen scores"
            p, f, b, wr = graded(recs, pick.get, lambda k, m: lab.get((k, m)), has)
            name = "with copy sums (shipping)" if copy_sums else "without copy sums"
            print(f"    {name:26s} precision {100 * p:6.2f}%  found {pct(f)}  binds {b:,}, wrong {wr}")
    top10 = lambda d: {w for w, _ in sorted(d.items(), key=lambda wp: -wp[1])[:10]}
    both = [k for k in xs["p"] if k in xs["p_earlier"] and k in weight]
    same_cut = sum((max(xs["p"][k].values()) >= MINT_BELOW) == (max(xs["p_earlier"][k].values()) >= MINT_BELOW) for k in both)
    same_top = sum(top10(xs["p"][k]) == top10(xs["p_earlier"][k]) for k in both)
    print(f"  the two XGBoost models on these records: same 0.01 cut for {same_cut:,} of {len(both):,}, same top-10 set for {same_top:,}")


def heldout_one_set():
    weight = {r["record_key"]: r["weight"] for r in rows("heldout/records.jsonl.gz")}
    qs, copies, ys = shortlists("heldout/qwen.jsonl.gz", "opus_pick")
    recs = [(k, weight.get(k, 1.0)) for k in qs]
    has = lambda k: any(y == 1 for y in ys[k].values())
    print("== Held-out records: 8,568 incoming records never trained on, with XGBoost's best candidate >= 0.01")
    print("   (label = Opus's single pick from the shortlist; the bind threshold is set here)")
    # the threshold: the lowest Qwen score at which binds are >= 99% right, keeping the most found
    tops = sorted(((max(p for _, p in qs[k]), k, w) for k, w in recs if qs[k]), reverse=True)
    right = bound = 0.0
    pos = sum(w * has(k) for k, w in recs)
    best = None
    for t, k, w in tops:
        work = max(qs[k], key=lambda wp: wp[1])[0]
        y = ys[k].get(work)
        if y is not None:
            bound += w
            right += w * (y == 1)
        if bound and right / bound >= 0.99 and (best is None or right / pos >= best[1]):
            best = (t, right / pos)
    print(f"  threshold for 99% precision: {best[0]:.6f} (production uses {TAU:.3f})")
    chosen = {}
    for copy_sums in (False, True):
        choose = lambda k: decide(qs.get(k, []), copies.get(k, {}), 1.0, False, copy_sums)[0] or None
        chosen[copy_sums] = {k: choose(k) for k, _ in recs}
        p, f, b, wr = graded(recs, chosen[copy_sums].get, lambda k, m: ys[k].get(m), has)
        name = "with copy sums (shipping)" if copy_sums else "without copy sums"
        print(f"  {name:28s} precision {100 * p:6.2f}%  found {pct(f)}  binds {b:,}, wrong {wr}")
    # single-pick labels say yes on one candidate only, so a bind to another copy of the same paper reads as wrong
    new_wrong = [k for k, _ in recs if chosen[True][k] and not chosen[False][k] and ys[k].get(chosen[True][k]) == 0]
    copy_of_pick = [k for k in new_wrong if any(ys[k].get(w) == 1 for w in copies[k].get(chosen[True][k], ()))]
    print(f"   of the {len(new_wrong)} binds copy sums add that the single-pick label calls wrong, {len(copy_of_pick)} join a copy of "
          f"the work Opus picked (the same paper)")


def heldout():
    if not os.path.exists(os.path.join(DATA, "heldout", "qwen_earlier.jsonl.gz")):
        return heldout_one_set()
    xs = {"p": collections.defaultdict(dict), "p_earlier": collections.defaultdict(dict)}
    for r in rows("heldout/xgboost.jsonl.gz"):
        for col, d in xs.items():
            if r[col] is not None:
                d[r["record_key"]][r["work"]] = r[col]
    weight = {r["record_key"]: r["weight"] for r in rows("heldout/records.jsonl.gz")}
    with open(os.path.join(DATA, "heldout", "over_cap.json")) as f:
        over = {o["record_key"] for o in json.load(f)}
    print("== Held-out records: incoming records never trained on whose best candidate scores 0.01 or more")
    print("   (label = Opus's single pick from the shortlist)")
    for title, col, qfile, cap in (("the shipping matcher (its XGBoost's cut and top 10, then Qwen)", "p", "heldout/qwen.jsonl.gz", over),
                                   ("for reference: the earlier XGBoost's cut and top 10, where the threshold was set", "p_earlier",
                                    "heldout/qwen_earlier.jsonl.gz", set())):
        qs, copies, ys = shortlists(qfile, "opus_pick")
        recs = [(k, weight.get(k, 1.0)) for k in ys if qs.get(k) or k in cap]   # the earlier set: the records that were scored
        has = lambda k: any(y == 1 for y in ys[k].values())
        # the lowest Qwen score at which binds are >= 99% right, keeping the most found
        tops = sorted(((max(p for _, p in qs[k]), k, w) for k, w in recs if qs.get(k) and k not in cap), reverse=True)
        right = bound = 0.0
        pos = sum(w * has(k) for k, w in recs)
        best = None
        for t, k, w in tops:
            y = ys[k].get(max(qs[k], key=lambda wp: wp[1])[0])
            if y is not None:
                bound += w
                right += w * (y == 1)
            if bound and right / bound >= 0.99 and (best is None or right / pos >= best[1]):
                best = (t, right / pos)
        print(f"  {title}: {len(recs):,} records; threshold for 99% precision {best[0]:.6f} (production uses {TAU:.3f})")
        chosen = {}
        for copy_sums in (False, True):
            chosen[copy_sums] = {k: decide(qs.get(k, []), copies.get(k, {}), max(xs[col][k].values()) if xs[col].get(k) else 1.0,
                                           k in cap, copy_sums)[0] for k, _ in recs}
            assert "MISSING" not in chosen[copy_sums].values(), "a record past the 0.01 cut has no Qwen scores"
            p, f, b, wr = graded(recs, chosen[copy_sums].get, lambda k, m: ys[k].get(m), has)
            name = "with copy sums (shipping)" if copy_sums else "without copy sums"
            print(f"    {name:26s} precision {100 * p:6.2f}%  found {pct(f)}  binds {b:,}, wrong {wr}")
        # single-pick labels say yes on one candidate only, so a bind to another copy of the same paper reads as wrong
        new_wrong = [k for k, _ in recs if chosen[True][k] and not chosen[False][k] and ys[k].get(chosen[True][k]) == 0]
        copy_of_pick = [k for k in new_wrong if any(ys[k].get(w) == 1 for w in copies[k].get(chosen[True][k], ()))]
        print(f"     of the {len(new_wrong)} binds copy sums add that the single-pick label calls wrong, {len(copy_of_pick)} join a copy "
              f"of the work Opus picked")


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    if "all" in what:
        what = ["nights", "baseline", "gold", "heldout"]
    for w in what:
        globals()[w]()
        print()

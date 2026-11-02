"""Qwen stage: shared tokenization for training and scoring. A row = prefix (the
record and its full shortlist) + one slot per candidate: the line ends ' =>' and a slot token ' ?'; the logits one position
before the slot must say ' yes' (Opus picked it) or ' no'. One forward pass scores every candidate, each slot sees the whole
shortlist, and no slot sees another slot's answer."""
import json
def ids_for(tok, row, maxlen=12000):
    T = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    q, yes, no = T(" ?"), T(" yes"), T(" no")
    assert len(q) == len(yes) == len(no) == 1, (q, yes, no)
    ids = T(row["prefix"]); slots = []
    for s in row["seats"]:
        t = T(s["line"] + " =>")
        if len(ids) + len(t) + 2 > maxlen: break
        ids += t; slots.append((len(ids), s["y"], s["key"])); ids += q + T("\n")
    return ids, slots, yes[0], no[0]
def read(path): return [json.loads(l) for l in open(path)]

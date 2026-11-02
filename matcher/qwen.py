"""Stage 2 scorer: Qwen3.5-9B with the released LoRA adapter, one forward pass per record, as production scores it.

Each candidate line ends with " =>" and a " ?" slot; P(yes) for that candidate = softmax over the " yes" / " no" logits at the
position before the slot. A record whose prompt reaches the 12,000-token cap before its first candidate fits is `over_cap`
(it mints). Needs a GPU with about 24 GB (bf16).
"""
BASE = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAXLEN = 12000


def ids_for(tok, row, maxlen=MAXLEN):
    T = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    q, yes, no = T(" ?"), T(" yes"), T(" no")
    assert len(q) == len(yes) == len(no) == 1, (q, yes, no)
    ids = T(row["prefix"]); slots = []
    for s in row["seats"]:
        t = T(s["line"] + " =>")
        if len(ids) + len(t) + 2 > maxlen:
            break
        ids += t; slots.append((len(ids), s["key"])); ids += q + T("\n")
    return ids, slots, yes[0], no[0]


class QwenScorer:
    def __init__(self, adapter, device="cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
        model = AutoModelForImageTextToText.from_pretrained(BASE, revision=BASE_REVISION, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to(device)
        self.model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()

    def score(self, row):
        """row = matcher.prompt.build(...) -> ([(key, p)], over_cap)"""
        torch = self.torch
        ids, slots, Y, N = ids_for(self.tok, row)
        if not slots:
            return [], True
        keep = torch.tensor([pos - 1 for pos, _ in slots], device=self.device)
        with torch.no_grad():
            lg = self.model(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=keep,
                            use_cache=False).logits[0][:, [Y, N]].float()
        return [(key, torch.softmax(two, -1)[0].item()) for (_, key), two in zip(slots, lg)], False

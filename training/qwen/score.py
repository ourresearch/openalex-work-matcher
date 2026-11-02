"""Qwen stage: p(yes) at every candidate slot, one forward pass per record (as production scores).
    python score.py --base Qwen/Qwen3.5-9B --adapter adapter --data rows.jsonl --out scores.jsonl --shard 0 --nshard 1"""
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel
from common import ids_for, read
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--adapter", default=""); ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--maxlen", type=int, default=12000); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(f"cuda:{a.shard % torch.cuda.device_count()}")
if a.adapter: model = PeftModel.from_pretrained(model, a.adapter); model = model.merge_and_unload()
model.eval(); dev = next(model.parameters()).device
rows = read(a.data)[a.shard::a.nshard]
with open(a.out, "w") as f, torch.no_grad():
    for k, r in enumerate(rows):
        ids, slots, Y, N = ids_for(tok, r, a.maxlen)
        keep = torch.tensor([pos - 1 for pos, _, _ in slots], device=dev)
        lg = model(input_ids=torch.tensor([ids], device=dev), logits_to_keep=keep).logits[0][:, [Y, N]].float()
        for (pos, y, key), two in zip(slots, lg):
            p = torch.softmax(two, -1)[0].item()
            f.write(json.dumps({"seed_id": r["seed_id"], "cid": key, "y": y, "p": p}) + "\n")
        if k % 20 == 0: print(f"row {k}/{len(rows)}", flush=True)
print("done", a.out, flush=True)

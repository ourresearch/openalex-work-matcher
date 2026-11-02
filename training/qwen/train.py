"""Qwen stage: LoRA training (accelerate DDP, one record per GPU per step). The shipped adapter:
    accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train.py --base Qwen/Qwen3.5-9B \
        --train rows_train.jsonl --out adapter --epochs 2 --lr 1e-4 --r 32 --acc 4
(8 GPUs x accumulation 4 = an effective batch of 32 records)."""
import argparse, math, os, random, time, torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from common import ids_for, read
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--acc", type=int, default=4)
ap.add_argument("--r", type=int, default=32); ap.add_argument("--maxlen", type=int, default=12000); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
acc = Accelerator(gradient_accumulation_steps=a.acc, mixed_precision="bf16")
tok = AutoTokenizer.from_pretrained(a.base)
rows = read(a.train); random.Random(0).shuffle(rows)
if a.limit: rows = rows[:a.limit]
ex = []
for r in rows:
    ids, slots, Y, N = ids_for(tok, r, a.maxlen)
    lab = [(pos - 1, y) for pos, y, _ in slots if y is not None]   # logits at pos-1 predict the slot
    if lab: ex.append((ids, lab))
acc.print(f"rows {len(rows)} -> examples {len(ex)}, tokens {sum(len(i) for i, _ in ex)}, labelled slots {sum(len(L) for _, L in ex)}, yes {sum(y for _, L in ex for _, y in L)}")
model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.config.use_cache = False
model.enable_input_require_grads()
cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)")
model = get_peft_model(model, cfg)
if acc.is_main_process: model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
per = math.ceil(len(ex) / acc.num_processes); steps = int(per * a.epochs / a.acc)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(1, steps))))
model, opt = acc.prepare(model, opt)   # scheduler stepped by hand once per optimizer step (accelerate's prepared scheduler steps per process)
model.train(); t0 = time.time(); step = 0; seen = 0; run_loss = 0.0; nb = 0
order = list(range(acc.process_index, len(ex), acc.num_processes))
ep = 0
while step < steps:
    random.Random(ep).shuffle(order); ep += 1
    for i in order:
        ids, lab = ex[i]
        x = torch.tensor([ids], device=acc.device); keep = torch.tensor([p for p, _ in lab], device=acc.device)
        y = torch.tensor([0 if v == 1 else 1 for _, v in lab], device=acc.device)   # class 0 = ' yes', 1 = ' no'
        with acc.accumulate(model):
            lg = model(input_ids=x, logits_to_keep=keep).logits[0][:, [Y, N]].float()   # only the slot rows: no 12K x 248K logits
            loss = torch.nn.functional.cross_entropy(lg, y)
            acc.backward(loss); run_loss += loss.item(); nb += 1
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if acc.sync_gradients:
            sched.step(); step += 1
            if step % 10 == 0 or step == 1:
                acc.print(f"step {step}/{steps} loss {run_loss / nb:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True); run_loss = 0.0; nb = 0
            if step >= steps: break
acc.wait_for_everyone()
if acc.is_main_process:
    acc.unwrap_model(model).save_pretrained(a.out); tok.save_pretrained(a.out); print("saved", a.out, flush=True)

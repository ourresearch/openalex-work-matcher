"""Qwen stage on Modal (one way to run train.py / score.py on 8 GPUs; any 8 x H100 / H200 machine works the same way).
Volume `/vol` holds code/{common,train,score}.py, data/rows_*.jsonl, models/<tag>, results/<tag>/<file>.scores.jsonl.
    modal deploy modal_app.py
    train8(tag, base, train_file, epochs, lr, r, score_files)  -> models/<tag> + results/<tag>/<file>.scores.jsonl
    score8(tag, base, adapter, files)                          -> results/<tag>/<file>.scores.jsonl (adapter "" = the base model)
The shipped adapter: train8(tag="adapter", base="Qwen/Qwen3.5-9B", train_file="rows_train.jsonl", epochs=2, lr=1e-4, r=32)."""
import modal, subprocess, os, sys, time, glob
app = modal.App("openalex-work-matcher-train")
vol = modal.Volume.from_name("openalex-work-matcher-train", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.14.1", "triton>=3.7.1", "transformers==5.18.0", "peft==0.21.1", "accelerate", "huggingface_hub[hf_xet]",
         "flash-linear-attention", "numpy", "pillow", "torchvision").env({"HF_HOME": "/root/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}))
def _sh(cmd, log):
    print("==", " ".join(cmd), flush=True); t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd="/vol/code", text=True)
        for line in p.stdout:
            f.write(line); f.flush()
            if line.startswith(("step", "rows", "saved", "trainable", "Traceback", "done")) or "Error" in line: print(line.rstrip()[:300], flush=True)
    r = p.wait(); vol.commit(); print(f"   exit {r} {time.time() - t0:.0f}s", flush=True)
    if r: print(open(log).read()[-4000:], flush=True); raise RuntimeError(f"failed {cmd[:3]}")
def _fetch(base):
    from huggingface_hub import snapshot_download; t0 = time.time(); snapshot_download(base, max_workers=32); print(f"fetched {base} {time.time() - t0:.0f}s", flush=True)
def _score(tag, base, adapter, files, ngpu):
    _fetch(base); os.makedirs(f"/vol/results/{tag}", exist_ok=True)
    for fn in files:
        procs = []
        for i in range(ngpu):
            out = f"/vol/results/{tag}/{fn}.scores.{i}.jsonl"
            cmd = [sys.executable, "score.py", "--base", base, "--data", f"/vol/data/{fn}", "--out", out, "--shard", str(i), "--nshard", str(ngpu)] + (["--adapter", adapter] if adapter else [])
            procs.append(subprocess.Popen(cmd, cwd="/vol/code", stdout=open(f"/vol/results/{tag}/{fn}.score.{i}.log", "w"), stderr=subprocess.STDOUT))
        rc = [p.wait() for p in procs]
        with open(f"/vol/results/{tag}/{fn}.scores.jsonl", "w") as f:
            for i in range(ngpu): f.write(open(f"/vol/results/{tag}/{fn}.scores.{i}.jsonl").read() if os.path.exists(f"/vol/results/{tag}/{fn}.scores.{i}.jsonl") else "")
        vol.commit(); print(f"scored {fn}: rc {rc}", flush=True)
        if any(rc): print(open(f"/vol/results/{tag}/{fn}.score.{rc.index(next(x for x in rc if x))}.log").read()[-3000:], flush=True)
def _train(tag, base, train_file, epochs, lr, r, score_files, limit, eff_batch=32):
    import torch
    vol.reload(); os.makedirs(f"/vol/results/{tag}", exist_ok=True); m = f"/vol/models/{tag}"; n = torch.cuda.device_count(); acc = max(1, eff_batch // n)
    print(f"{n} GPUs, grad accumulation {acc} (effective batch {n * acc})", flush=True)
    if not os.path.exists(f"{m}/adapter_config.json"):
        _fetch(base); _sh([sys.executable, "-m", "accelerate.commands.launch", *(["--multi_gpu"] if n > 1 else []), "--num_processes", str(n), "--mixed_precision", "bf16", "train.py", "--base", base,
             "--train", f"/vol/data/{train_file}", "--out", m, "--epochs", str(epochs), "--lr", str(lr), "--r", str(r), "--limit", str(limit), "--acc", str(acc)], f"/vol/results/{tag}/train.log")
    if score_files: _score(tag, base, m, score_files.split(","), n)
    return "ok"
@app.function(image=image, gpu=["H200:8", "H100:8"], volumes={"/vol": vol}, timeout=10800, memory=256 * 1024, cpu=32)
def train8(tag: str, base: str, train_file: str, epochs: float = 1.0, lr: float = 1e-4, r: int = 32, score_files: str = "", limit: int = 0):
    return _train(tag, base, train_file, epochs, lr, r, score_files, limit)
@app.function(image=image, gpu=["H200", "H100"], volumes={"/vol": vol}, timeout=2400, memory=96 * 1024, cpu=8)
def train1(tag: str, base: str, train_file: str, epochs: float = 1.0, limit: int = 0):
    """Timing / smoke run on one GPU."""
    return _train(tag, base, train_file, epochs, 1e-4, 32, "", limit, eff_batch=8)
@app.function(image=image, gpu=["H200:8", "H100:8"], volumes={"/vol": vol}, timeout=3600, memory=256 * 1024, cpu=32)
def score8(tag: str, base: str, adapter: str, files: str):
    import torch
    vol.reload(); _score(tag, base, adapter, files.split(","), torch.cuda.device_count()); return "ok"

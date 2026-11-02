"""The embedding channel: Qwen3-Embedding-0.6B vectors of "Title / Abstract / Venue" (2,000 characters, no instruction), the
recipe OpenAlex embeds every work with. Production searches its own index of all works; here the public API's semantic search
proposes the neighbours and this module scores them, so emb_score is the same cosine production computes."""
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MAX_CHARS = 2000


def record_text(rec):
    t = rec.get("title")
    if t is None or not t.strip():
        return None
    s = "Title: " + t
    if rec.get("abstract") is not None:
        s += "\n\nAbstract: " + rec["abstract"]
    if rec.get("source_name") is not None:
        s += "\n\nVenue: " + rec["source_name"]
    return s[:MAX_CHARS]


def work_text(w):
    s = "Title: " + (w.get("title") or "")
    if w.get("abstract") is not None:
        s += "\n\nAbstract: " + w["abstract"]
    v = (w.get("primary_location") or {}).get("source_name")
    if v is not None:
        s += "\n\nVenue: " + v
    return s[:MAX_CHARS]


class Embedder:
    def __init__(self, device=None, batch=16):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION, padding_side="left")
        self.model = AutoModel.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION, dtype=torch.float32).to(self.device).eval()
        self.batch = batch

    def __call__(self, texts):
        """L2-normalised vectors (last-token pooling, as the model card specifies)."""
        torch, out = self.torch, []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self.tok(texts[i:i + self.batch], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, -1]
                out += torch.nn.functional.normalize(h.float(), p=2, dim=1).cpu().tolist()
        return out


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))

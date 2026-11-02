"""Download the release files too big for git, check their sha256 and unpack the Qwen adapter.

    python3 tools/download.py qwen-adapter.tar --to assets/          # -> assets/qwen-adapter/
    python3 tools/download.py all --to assets/

Files and checksums: models/MANIFEST.json ("assets") and data/RELEASE_ASSETS.json. Standard library only; resumes a partial
download."""
import argparse
import hashlib
import json
import os
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "ourresearch/openalex-work-matcher"


def manifest():
    m = json.load(open(os.path.join(ROOT, "models", "MANIFEST.json")))
    files = {a["file"]: a for a in m["assets"]}
    for a in json.load(open(os.path.join(ROOT, "data", "RELEASE_ASSETS.json"))):
        files.setdefault(a["file"], a)
    return m["release"], files


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(tag, a, to):
    path = os.path.join(to, a["file"])
    if os.path.exists(path) and os.path.getsize(path) == a["bytes"] and sha256(path) == a["sha256"]:
        print(f"{a['file']}: already here, checksum ok")
        return path
    url = f"https://github.com/{REPO}/releases/download/{tag}/{a['file']}"
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have >= a["bytes"]:
        have = 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(path, "ab" if have and r.status == 206 else "wb") as f:
        while b := r.read(1 << 20):
            f.write(b)
    got = sha256(path)
    if got != a["sha256"]:
        raise SystemExit(f"{a['file']}: sha256 {got} != {a['sha256']} (delete it and run again)")
    print(f"{a['file']}: downloaded, checksum ok")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="release file names, or all")
    p.add_argument("--to", default="assets")
    args = p.parse_args()
    tag, files = manifest()
    os.makedirs(args.to, exist_ok=True)
    for name in (list(files) if args.files == ["all"] else args.files):
        if name not in files:
            raise SystemExit(f"unknown file {name}; release files: {', '.join(files)}")
        path = fetch(tag, files[name], args.to)
        if name.endswith(".tar"):
            with tarfile.open(path) as t:
                # the "data" filter refuses unsafe paths; older Pythons lack it (the archive holds two plain files)
                t.extractall(args.to, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
            print(f"  unpacked into {os.path.join(args.to, name[:-4])}/")


if __name__ == "__main__":
    main()

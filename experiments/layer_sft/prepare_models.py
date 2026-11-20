#!/usr/bin/env python3
"""Explicitly download pinned, public model files into an isolated experiment.

Run this on the authorized training host, before the offline trainer. No private
examples, prompts, or answers are sent to a model provider. No remote Python model
code is admitted. A populated target is verified and reused, never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(spec_path):
    from huggingface_hub import HfApi, snapshot_download

    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    revision, model_id = spec["revision"], spec["model_id"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Pin an exact 40-character model commit before downloading")
    root = spec_path.parent
    dest = root / spec["snapshot_path"]
    if dest.is_symlink():
        raise ValueError("Model target must not be a symlink")
    dest = dest.resolve()
    if root not in dest.parents:
        raise ValueError("Model destination must be within the experiment directory")
    lock_path = dest / "snapshot-lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if (lock["model_id"], lock["revision"]) != (model_id, revision):
            raise ValueError("Existing model belongs to a different pinned revision")
        for name, digest in lock["files"].items():
            p = (dest / name).resolve()
            if dest not in p.parents or p.is_symlink() or sha256(p) != digest:
                raise ValueError("Existing model file verification failed: " + name)
        print(json.dumps({"status": "verified_existing", "model": model_id,
                          "revision": revision}, ensure_ascii=False), flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    if shutil.disk_usage(dest).free < 20 * 1024**3:
        raise RuntimeError("At least 20 GiB of free experiment-disk space is required")
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError("Model provider resolved a different revision")
    suffixes = (".json", ".jinja", ".txt", ".model", ".safetensors")
    allowed = [s.rfilename for s in info.siblings
               if "/" not in s.rfilename and
               (s.rfilename.endswith(suffixes) or s.rfilename in {"LICENSE", "README.md"})]
    if not any(n.endswith(".safetensors") for n in allowed):
        raise RuntimeError("No safe tensor weights were found in the pinned model")
    # huggingface_hub authenticates downloads; execute no checkpoint-provided code.
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(dest),
                      allow_patterns=allowed, max_workers=2)
    files = {name: sha256(dest / name) for name in sorted(allowed)}
    for sibling in info.siblings:
        if sibling.rfilename not in files:
            continue
        p = dest / sibling.rfilename
        if sibling.size is not None and p.stat().st_size != sibling.size:
            raise RuntimeError("Downloaded file size mismatch: " + sibling.rfilename)
        lfs = getattr(sibling, "lfs", None)
        expected = getattr(lfs, "sha256", None) if lfs is not None else None
        if isinstance(lfs, dict):
            expected = lfs.get("sha256")
        if expected and files[sibling.rfilename] != expected:
            raise RuntimeError("Provider LFS hash mismatch: " + sibling.rfilename)
    config = json.loads((dest / "config.json").read_text())
    if config.get("auto_map"):
        raise RuntimeError("This experiment does not execute custom model code")
    lock = {"model_id": model_id, "revision": revision, "files": files,
            "provider_sizes_checked": True, "lfs_sha256_checked_when_present": True,
            "trust_remote_code": False}
    temporary = lock_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, lock_path)
    print(json.dumps({"status": "downloaded_verified", "model": model_id,
                      "revision": revision, "file_count": len(files)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("specs", nargs="+", type=Path)
    args = p.parse_args()
    for spec_file in args.specs:
        prepare(spec_file)

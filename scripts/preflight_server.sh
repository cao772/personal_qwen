#!/usr/bin/env bash
# Generic read-only inspection of the CURRENT host; no remote access or writes.
set -euo pipefail
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' 'ERROR: python3 is required for read-only preflight.' >&2
  exit 2
fi
exec python3 -I -B - "$@" <<'PY'
import argparse
import csv
import io
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import stat
import subprocess
import sys
from datetime import datetime, timezone


def run_readonly(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
        return {"ok": result.returncode == 0, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}


def csv_rows(result, fields):
    if not result.get("ok"):
        return result
    rows = []
    for row in csv.reader(io.StringIO(result["stdout"]), skipinitialspace=True):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(fields):
            return {"ok": False, "error": "Unexpected nvidia-smi CSV format",
                    "raw_output": result["stdout"][:4000]}
        rows.append(dict(zip(fields, (cell.strip() for cell in row))))
    return {"ok": True, "rows": rows}


def disk_info(path):
    item = {"path": str(path), "exists": path.exists()}
    if not item["exists"]:
        return item
    try:
        info = os.statvfs(path)
        block_size = info.f_frsize or info.f_bsize
        used = (info.f_blocks - info.f_bfree) * block_size
        available = info.f_bavail * block_size
        item.update({"total_gib": round(info.f_blocks * block_size / 2**30, 2),
                     "available_to_current_user_gib": round(available / 2**30, 2),
                     "used_percent_of_used_plus_available": round(100 * used / (used + available), 2)
                         if used + available else None})
    except OSError as error:
        item["error"] = str(error)
    return item


def path_info(path):
    item = {"path": str(path), "exists": path.exists()}
    if not item["exists"]:
        return item
    try:
        info = path.stat()
        item.update({"is_directory": stat.S_ISDIR(info.st_mode), "owner_uid": info.st_uid,
                     "owned_by_current_user": info.st_uid == os.geteuid(),
                     "mode": oct(stat.S_IMODE(info.st_mode)),
                     "readable": os.access(path, os.R_OK),
                     "writable": os.access(path, os.W_OK), "size_bytes": info.st_size})
    except OSError as error:
        item["error"] = str(error)
    return item


parser = argparse.ArgumentParser(description=(
    "Read-only preflight on the current host; does not connect to any remote host. "
    "Only explicitly supplied model paths are inspected, using fixed filename metadata."))
parser.add_argument("--workspace", type=Path, help="Optional existing intended workspace to inspect")
parser.add_argument("--model-dir", type=Path, action="append", default=[],
                    help="Explicit local model directory; repeat as needed; no recursive search")
parser.add_argument("--python", default=sys.executable,
                    help="Existing Python interpreter whose installed package metadata is queried")
args = parser.parse_args()
user = pwd.getpwuid(os.geteuid())
report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "read_only": True,
          "host": socket.gethostname(), "username": user.pw_name, "uid": os.geteuid(),
          "kernel": list(os.uname()), "current_directory": os.getcwd(),
          "scope": "Current host only; no remote connection, installs, GPU model load, or file writes."}
smi = shutil.which("nvidia-smi")
if smi:
    fields = ["index", "uuid", "name", "memory.total", "memory.used", "memory.free", "utilization.gpu"]
    report["gpus"] = csv_rows(run_readonly([smi, "--query-gpu=" + ",".join(fields),
                                          "--format=csv,noheader,nounits"]), fields)
    fields = ["gpu_uuid", "used_gpu_memory"]
    apps = csv_rows(run_readonly([smi, "--query-compute-apps=" + ",".join(fields),
                                 "--format=csv,noheader,nounits"]), fields)
    if apps.get("ok"):
        grouped = {}
        for app in apps["rows"]:
            summary = grouped.setdefault(app["gpu_uuid"], {"process_count": 0,
                                                           "known_used_memory_mib": 0,
                                                           "unknown_memory_entries": 0})
            summary["process_count"] += 1
            memory = app["used_gpu_memory"]
            if memory.isdigit():
                summary["known_used_memory_mib"] += int(memory)
            else:
                summary["unknown_memory_entries"] += 1
        report["compute_apps"] = {"ok": True, "by_gpu_uuid": grouped,
                                  "privacy": "Only aggregate counts/memory; no names, PIDs, command lines, or owners."}
    else:
        report["compute_apps"] = apps
else:
    report["gpus"] = report["compute_apps"] = {"ok": False, "error": "nvidia-smi is unavailable"}
report["gpu_units"] = {"memory": "MiB", "utilization": "percent"}
report["idle_note"] = ("utilization=0 does not establish availability. The runner also checks all "
                       "GPU1 XML processes, residual memory, and a second observation before launch.")

disk_paths = [Path("/"), Path(user.pw_dir), Path("/home"), Path("/data")]
if args.workspace is not None:
    workspace = args.workspace.expanduser().resolve()
    report["workspace"] = path_info(workspace)
    disk_paths.append(workspace)
report["disk_space"] = [disk_info(path) for path in dict.fromkeys(disk_paths)]

dependency_probe = r'''
import importlib.metadata as metadata
import json, sys
names = ["numpy", "torch", "transformers", "safetensors", "huggingface-hub", "accelerate", "peft"]
packages = {}
for name in names:
    try:
        packages[name] = {"installed": True, "version": metadata.version(name)}
    except metadata.PackageNotFoundError:
        packages[name] = {"installed": False}
print(json.dumps({"executable": sys.executable, "python_version": sys.version,
                  "packages": packages, "gpu_libraries_imported": False}))
'''
python_result = run_readonly([args.python, "-I", "-B", "-c", dependency_probe])
if python_result.get("ok"):
    try:
        report["python_environment"] = json.loads(python_result["stdout"])
    except json.JSONDecodeError:
        report["python_environment"] = {"ok": False, "error": "Unexpected interpreter output"}
else:
    report["python_environment"] = python_result

report["model_paths"] = []
fixed_names = ["config.json", "model.safetensors", "model.safetensors.index.json",
               "tokenizer.json", "tokenizer_config.json", "generation_config.json"]
for supplied in args.model_dir:
    directory = supplied.expanduser().resolve()
    model = path_info(directory)
    model["inspection"] = "Metadata of exact filenames only; no file contents or recursive search."
    model["files"] = []
    if model.get("is_directory") and model.get("readable"):
        for filename in fixed_names:
            model["files"].append(path_info(directory / filename))
    report["model_paths"].append(model)
if not args.model_dir:
    report["model_path_note"] = "No --model-dir supplied; no model locations were searched."

print(json.dumps(report, ensure_ascii=False, indent=2))
sys.exit(0 if report["gpus"].get("ok") and report["compute_apps"].get("ok")
         and python_result.get("ok") else 1)
PY

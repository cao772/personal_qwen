#!/usr/bin/env bash
# Foreground, opt-in launcher. No SSH, installs, process termination, or sudo.
set -euo pipefail
umask 077
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' 'ERROR: python3 is required to verify GPU availability.' >&2
  exit 2
fi
# Preserve the caller's stdin while Python reads this embedded program.
exec 3<&0
exec python3 -I -B - "$@" <<'PY'
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import pwd
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

MAX_IDLE_USED_MIB = 512
MAX_DISK_USED_PERCENT = 95


class Blocked(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise Blocked(reason)


def smi_query(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Blocked("Unable to verify GPU state: " + str(error)) from error
    require(result.returncode == 0,
            "nvidia-smi query failed; GPU availability is unverified: " + result.stderr.strip()[:1000])
    require(len(result.stdout) <= 4 * 1024 * 1024, "Unexpectedly large nvidia-smi response")
    return result.stdout.strip()


def csv_records(text, width):
    rows = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or not any(cell.strip() for cell in row):
            continue
        require(len(row) == width, "Unexpected nvidia-smi CSV; availability is unverified")
        rows.append([cell.strip() for cell in row])
    return rows


def unsigned(value, label):
    require(re.fullmatch(r"\d+", value) is not None, label + " is unavailable or nonnumeric")
    return int(value)


def gpu_probe(smi, selected_uuid=None):
    selector = selected_uuid if selected_uuid else "1"
    fields = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode"
    rows = csv_records(smi_query([smi, "-i", selector, "--query-gpu=" + fields,
                                   "--format=csv,noheader,nounits"]), 8)
    require(len(rows) == 1, "Physical GPU index 1 cannot be uniquely identified")
    index, uuid, name, total, used, free, util, mode = rows[0]
    require(index == "1", "GPU UUID no longer maps to physical index 1")
    require(re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid) is not None, "Invalid/unavailable GPU UUID")
    require(selected_uuid is None or uuid == selected_uuid, "GPU UUID changed between observations")
    require(mode not in ("Prohibited", "N/A", "[N/A]", "[Not Supported]"), "GPU compute mode is unavailable/prohibited")
    state = {"physical_index": 1, "uuid": uuid, "name": name,
             "total_mib": unsigned(total, "total memory"),
             "used_mib": unsigned(used, "used memory"),
             "free_mib": unsigned(free, "free memory"),
             "utilization_percent": unsigned(util, "GPU utilization"), "compute_mode": mode}
    require(state["total_mib"] > 0 and state["free_mib"] <= state["total_mib"]
            and state["used_mib"] <= state["total_mib"]
            and state["used_mib"] + state["free_mib"] <= state["total_mib"] + 2
            and state["utilization_percent"] <= 100, "Inconsistent GPU memory/utilization counters")
    apps_text = smi_query([smi, "-i", uuid, "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                          "--format=csv,noheader,nounits"])
    apps = csv_records(apps_text, 3)
    state["compute_apps"] = []
    for app_uuid, pid, memory in apps:
        require(app_uuid == uuid, "Compute-app query returned a different GPU")
        state["compute_apps"].append({"pid": pid, "used_gpu_memory_mib": memory})
    xml = smi_query([smi, "-i", uuid, "-q", "-x"])
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise Blocked("GPU process XML could not be parsed; availability is unverified") from error
    gpus = root.findall("gpu")
    require(len(gpus) == 1 and gpus[0].findtext("uuid", "").strip() == uuid,
            "GPU process XML does not identify the selected UUID")
    processes = gpus[0].find("processes")
    require(processes is not None, "GPU process list is missing; availability is unverified")
    require(not (processes.text or "").strip(), "GPU process list is unavailable or unsupported")
    state["all_gpu_processes"] = []
    for process in list(processes):
        require(process.tag == "process_info", "Unknown GPU process-list element")
        pid = process.findtext("pid", "").strip()
        require(pid.isdigit(), "GPU process PID is unavailable")
        state["all_gpu_processes"].append({"pid": int(pid), "type": process.findtext("type", "unknown"),
                                            "used_memory": process.findtext("used_memory", "unknown")})
    return state


def require_idle(state, min_free_gib):
    require(not state["compute_apps"] and not state["all_gpu_processes"],
            "GPU1 has listed processes; no process will be stopped or ignored")
    require(state["used_mib"] <= MAX_IDLE_USED_MIB,
            "GPU1 still uses %d MiB (idle ceiling %d MiB). Residual/defunct allocations are not assumed safe."
            % (state["used_mib"], MAX_IDLE_USED_MIB))
    require(state["free_mib"] >= math.ceil(min_free_gib * 1024), "GPU1 free memory is below the requested reserve")
    require(state["utilization_percent"] == 0, "GPU1 is active despite empty process lists; retry after verifying availability")


def workspace_check(path, min_free_gib, uid):
    try:
        workspace = path.expanduser().resolve(strict=True)
        info = workspace.stat()
    except OSError as error:
        raise Blocked("Explicit workspace must already exist: " + str(error)) from error
    require(stat.S_ISDIR(info.st_mode), "Workspace is not a directory")
    require(info.st_uid == uid, "Workspace must be owned by the current user")
    require(os.access(workspace, os.W_OK | os.X_OK), "Workspace is not writable/searchable by the current user")
    disk = os.statvfs(workspace)
    block_size = disk.f_frsize or disk.f_bsize
    used, available = (disk.f_blocks - disk.f_bfree) * block_size, disk.f_bavail * block_size
    percent = 100 * used / (used + available) if used + available else 100
    require(available >= math.ceil(min_free_gib * 2**30), "Workspace filesystem has insufficient available space")
    require(percent < MAX_DISK_USED_PERCENT,
            "Workspace filesystem is %.2f%% used; the runner refuses >= %d%% and will not choose another disk"
            % (percent, MAX_DISK_USED_PERCENT))
    return workspace, {"path": str(workspace), "owner_uid": info.st_uid,
                       "available_gib": round(available / 2**30, 2), "used_percent": round(percent, 2)}


def safe_subdir(workspace, relative, uid):
    target = workspace / relative
    resolved = target.resolve()
    require(workspace in resolved.parents, "Cache/temp path escapes the explicit workspace")
    if target.exists() or target.is_symlink():
        info = target.stat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == uid
                and os.access(target, os.W_OK | os.X_OK), "Cache/temp path is not a directory owned by the current user")
    else:
        target.mkdir(mode=0o700)
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=(
        "Verify physical GPU1, then optionally run one foreground command bound to its UUID. "
        "Default is read-only dry-run. This is a launcher, not a cluster scheduler or sandbox."))
    parser.add_argument("--execute", action="store_true", help="Actually launch the supplied command after checks")
    parser.add_argument("--workspace", required=True, type=Path, help="Explicit existing directory owned by current user")
    parser.add_argument("--expected-user", required=True, help="Explicit expected local username; no user is assumed")
    parser.add_argument("--expected-host", required=True, help="Explicit expected hostname; no host is assumed")
    parser.add_argument("--min-gpu-free-gib", type=float, default=20.0)
    parser.add_argument("--min-disk-free-gib", type=float, default=20.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command and arguments after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(bool(command), "Supply a command after --; no default training command is invented")
    for value in (args.min_gpu_free_gib, args.min_disk_free_gib):
        require(math.isfinite(value) and value > 0, "Free-space thresholds must be positive finite GiB values")
    uid = os.geteuid()
    require(uid != 0 and os.getuid() != 0, "Do not run this launcher as root; no privilege escalation is supported")
    user = pwd.getpwuid(uid).pw_name
    host = socket.gethostname()
    require(user == args.expected_user, "Current user does not match --expected-user")
    require(host.split(".")[0].casefold() == args.expected_host.split(".")[0].casefold(),
            "Current host does not match --expected-host")
    name = Path(command[0]).name
    require(name not in {"sudo", "su", "doas", "pkexec", "runuser", "kill", "killall", "pkill",
                         "nvidia-smi", "systemctl", "service", "reboot", "shutdown", "poweroff", "vllm"},
            "Privilege/process/device/service-management commands are not supported by this training launcher")
    require(not any(command[i] == "-m" and (command[i + 1] == "vllm" or command[i + 1].startswith("vllm."))
                    for i in range(len(command) - 1)), "This launcher does not start vLLM services")
    workspace, disk = workspace_check(args.workspace, args.min_disk_free_gib, uid)
    if "/" in command[0]:
        executable = Path(command[0]).expanduser()
        if not executable.is_absolute():
            executable = workspace / executable
        executable = str(executable.resolve())
    else:
        executable = shutil.which(command[0])
    require(executable is not None and os.path.isfile(executable) and os.access(executable, os.X_OK),
            "Command executable is missing or not executable")
    command[0] = executable
    smi = shutil.which("nvidia-smi")
    require(smi is not None, "nvidia-smi is unavailable; GPU state cannot be verified")
    first = gpu_probe(smi)
    require_idle(first, args.min_gpu_free_gib)
    time.sleep(1)
    second = gpu_probe(smi, first["uuid"])
    require_idle(second, args.min_gpu_free_gib)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "host": host, "user": user,
              "execute": args.execute, "workspace": disk, "gpu": second,
              "environment_binding": {"CUDA_VISIBLE_DEVICES": second["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
              "inside_process_device": "cuda:0 is the selected physical GPU1; do not select cuda:1",
              "command": {"executable": executable, "argument_count": len(command) - 1},
              "policy": {"max_idle_used_mib": MAX_IDLE_USED_MIB,
                         "min_gpu_free_gib": args.min_gpu_free_gib,
                         "min_disk_free_gib": args.min_disk_free_gib,
                         "max_disk_used_percent_exclusive": MAX_DISK_USED_PERCENT},
              "reservation_note": "Checks are observations, not an exclusive cross-user GPU reservation."}
    if not args.execute:
        report["result"] = "DRY_RUN: verified; no directories, locks, caches, or training process were created."
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    # Only the explicit --execute branch can write or launch a command.
    os.umask(0o077)
    lock_path = workspace / (".gpu1-" + second["uuid"] + ".lock")
    require(not lock_path.is_symlink(), "Workspace lock must not be a symlink")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    lock_stat = os.fstat(lock_fd)
    require(stat.S_ISREG(lock_stat.st_mode) and lock_stat.st_uid == uid, "Workspace lock must be a regular file owned by current user")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise Blocked("Another cooperating run already holds this workspace GPU1 lock") from error
    os.set_inheritable(lock_fd, True)
    # Cooperating runs lock only this workspace. Check hardware again immediately
    # before launch; a job started by another user is still outside this lock.
    final = gpu_probe(smi, second["uuid"])
    require_idle(final, args.min_gpu_free_gib)
    workspace_check(workspace, args.min_disk_free_gib, uid)
    cache = safe_subdir(workspace, ".cache", uid)
    hf = safe_subdir(workspace, ".cache/huggingface", uid)
    torch_cache = safe_subdir(workspace, ".cache/torch", uid)
    temp = safe_subdir(workspace, ".tmp", uid)
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": final["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "XDG_CACHE_HOME": cache, "HF_HOME": hf, "HF_HUB_CACHE": hf + "/hub",
                        "HUGGINGFACE_HUB_CACHE": hf + "/hub", "HF_DATASETS_CACHE": hf + "/datasets",
                        "TORCH_HOME": torch_cache, "TMPDIR": temp, "PYTHONDONTWRITEBYTECODE": "1"})
    # Inherited legacy cache paths must not silently route checkpoint downloads
    # onto another filesystem, such as an already-full /data volume.
    environment.pop("TRANSFORMERS_CACHE", None)
    environment.pop("PYTORCH_TRANSFORMERS_CACHE", None)
    environment.pop("PYTORCH_PRETRAINED_BERT_CACHE", None)
    report["gpu"] = final
    report["result"] = "Launching one foreground command; no process termination or background service."
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    os.chdir(workspace)
    os.dup2(3, 0)
    os.close(3)
    os.execvpe(command[0], command, environment)


try:
    sys.exit(main())
except (Blocked, OSError) as error:
    print("BLOCKED: " + str(error), file=sys.stderr, flush=True)
    sys.exit(2)
PY

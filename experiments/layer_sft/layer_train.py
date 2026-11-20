#!/usr/bin/env python3
"""Local, explicit-layer SFT; an execution record is not a quality verdict.

No network, installation, pruning, LoRA, service changes, or robot actions.
Only reviewed final answers receive loss. Earlier assistant replies are context.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RESEARCH_KINDS = {"grounded_reading", "grounded_correction", "synthetic_reading", "replay_synthetic"}
SUPPORTED = {"openbmb/MiniCPM5-2B", "Qwen/Qwen3-1.7B"}
ACTIVE_REPORT_PATH = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resolve_path(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def verify_snapshot(folder, model_id, revision):
    require(model_id in SUPPORTED, "Unreviewed model family")
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "A full immutable model revision is required")
    require(folder.is_dir(), "Model snapshot is missing; prepare it separately")
    lock_path = folder / "snapshot-lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        require(lock["model_id"] == model_id and lock["revision"] == revision,
                "Snapshot lock does not match model and revision")
        require(isinstance(lock["files"], dict) and lock["files"], "Empty snapshot lock")
        files = {}
        for name, digest in lock["files"].items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts
                    and name != "snapshot-lock.json", "Unsafe snapshot lock path")
            require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid locked SHA-256")
            path = folder / relative
            require(path.is_file(), "Locked model asset is missing: " + name)
            files[name] = sha(path)
            require(files[name] == digest, "Snapshot bytes differ from lock: " + name)
        for path in folder.iterdir():
            if path.is_file() and path.suffix in {".json", ".jinja", ".safetensors", ".model", ".txt"}:
                require(path.name == "snapshot-lock.json" or path.name in files,
                        "An unlocked runtime asset is present: " + path.name)
    else:
        require(folder.name == revision and folder.parent.name == "snapshots",
                "Use a revision-named HF cache snapshot or a verified snapshot-lock.json")
        files = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()
                 and p.suffix in {".json", ".jinja", ".safetensors", ".model", ".txt"}}
    require({"config.json", "tokenizer_config.json"} <= set(files), "Required model/tokenizer configuration is missing")
    require(any(name in files for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt")),
            "Tokenizer vocabulary is missing")
    if "model.safetensors.index.json" in files:
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        require(bool(index.get("weight_map")), "Empty safetensors index")
        require(set(index["weight_map"].values()) <= set(files), "The safetensors index names an absent or unlocked shard")
    else:
        require("model.safetensors" in files, "Safetensors weights or their index are missing")
    return files


def load_manifest(folder, evaluation_files):
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = ["train", "dev"] + [Path(name).stem for name in evaluation_files]
    require(len(names) == len(set(names)), "Repeated dataset split")
    groups = {}
    for name in names:
        info = manifest["splits"][name]
        require(isinstance(info["group_ids"], list), "Manifest must freeze split group_ids")
        groups[name] = set(info["group_ids"])
        require(sha(folder / (name + ".jsonl")) == info["sha256"], "Dataset fingerprint mismatch: " + name)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            require(not groups[name] & groups[other], "Dataset group overlap: " + name + "/" + other)
    return manifest


def read_split(folder, manifest, name):
    info = manifest["splits"][name]
    path = folder / (name + ".jsonl")
    require(sha(path) == info["sha256"], "Dataset changed after manifest verification")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == info["rows"], "Dataset row-count mismatch")
    require({r["group_id"] for r in rows} == set(info["group_ids"]), "Dataset group manifest mismatch")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate example identifiers")
    return rows


def admit_row(row, feedback_intake_sha256=None):
    messages = row["messages"]
    require(len(messages) >= 2 and messages[-1].get("role") == "assistant"
            and messages[-2].get("role") == "user", "Expected a final assistant answer after a user turn")
    require(all(m.get("role") in {"system", "user", "assistant"}
                and isinstance(m.get("content"), str) for m in messages), "Unsupported message shape")
    require(row.get("loss_on") == "last_assistant_only"
            and row.get("target_message_index") == len(messages) - 1,
            "Only the final assistant response may be supervised")
    if row.get("kind") == "user_approved_feedback":
        require(row.get("user_approved") is True and feedback_intake_sha256
                and row.get("admission_intake_sha256") == feedback_intake_sha256,
                "Feedback must match the freshly verified intake snapshot")
    else:
        require(row.get("kind") in RESEARCH_KINDS and row.get("user_approved") is False
                and row.get("target_status") == "researcher_written_not_user_approved",
                "Unreviewed or incorrectly attributed training target")
        if row["kind"].startswith("grounded_"):
            require(bool(row.get("sources")), "Grounded example has no original source")
    require(bool(messages[-1]["content"].strip()), "Empty final target")


def encode_final_answer(tokenizer, row, max_tokens, feedback_intake_sha256=None):
    """Render the complete official template and mask everything before the final answer.

    Existing targets contain no reasoning trace. Any empty thinking block inserted
    by the template is MASKED CONTEXT, not an approved reasoning demonstration.
    Native-thinking generation must be evaluated separately after training.
    """
    admit_row(row, feedback_intake_sha256)
    messages = [dict(m) for m in row["messages"]]
    require(all(not m.get("reasoning_content") and not m.get("reasoning") for m in messages),
            "Reasoning targets require a separately reviewed encoding recipe")
    answer = messages[-1]["content"]
    sentinel = "FINAL_RESPONSE_TARGET_" + fingerprint(messages)
    require(all(sentinel not in m["content"] for m in messages), "Sentinel collision")
    marked = [dict(m) for m in messages]
    marked[-1]["content"] = sentinel
    template_args = dict(tokenize=False, add_generation_prompt=False, enable_thinking=True)
    marked_text = tokenizer.apply_chat_template(marked, **template_args)
    require(marked_text.count(sentinel) == 1, "Template does not expose one final answer region")
    prefix, suffix = marked_text.split(sentinel)
    full_text = tokenizer.apply_chat_template(messages, **template_args)
    require(full_text == prefix + answer + suffix, "Template transformed the final answer unexpectedly")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    require(ids[:len(prefix_ids)] == prefix_ids, "Tokenizer merged across the answer boundary")
    require(0 < len(prefix_ids) < len(ids) <= max_tokens, "Empty/overlong example; truncation is forbidden")
    labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
    return {"id": row["id"], "kind": row["kind"], "input_ids": ids, "labels": labels,
            "target_tokens": len(ids) - len(prefix_ids), "masked_tokens": len(prefix_ids),
            "prior_assistant_turns": sum(m["role"] == "assistant" for m in messages[:-1]),
            "template_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest()}


def selected_parameters(model, last_layers=2):
    require(model.config.model_type == "llama", "Direct layer update is restricted to reviewed Llama architecture")
    count = len(model.model.layers)
    require(0 < last_layers <= count, "Invalid layer count")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = {}
    for name, parameter in model.named_parameters():
        match = re.fullmatch(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight", name)
        if match and int(match[1]) >= count - last_layers and match[2] in PROJECTIONS:
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            selected[name] = parameter
    require(len(selected) == last_layers * len(PROJECTIONS), "Unexpected projection names/count; stop before training")
    return selected


def tensor_sha(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def frozen_fingerprints(model, selected):
    return {name: tensor_sha(parameter) for name, parameter in model.named_parameters() if name not in selected}


def copy_state(selected):
    return {name: parameter.detach().cpu().clone().contiguous() for name, parameter in selected.items()}


def apply_state(selected, state):
    import torch
    require(set(selected) == set(state), "Saved parameter names do not match the target layers")
    with torch.no_grad():
        for name, parameter in selected.items():
            require(tuple(parameter.shape) == tuple(state[name].shape), "Saved parameter shape mismatch")
            require(torch.isfinite(state[name]).all().item(), "Saved nonfinite parameter")
            parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))


def batch(rows, pad_token_id, device):
    import torch
    width = max(len(row["input_ids"]) for row in rows)
    return {"input_ids": torch.tensor([r["input_ids"] + [pad_token_id] * (width - len(r["input_ids"])) for r in rows], device=device),
            "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows], device=device),
            "labels": torch.tensor([r["labels"] + [-100] * (width - len(r["labels"])) for r in rows], device=device)}


def amp_context(device):
    import torch
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()


def dev_loss(model, rows, pad, device):
    import torch
    model.eval()
    totals = collections.defaultdict(lambda: [0.0, 0])
    with torch.no_grad(), amp_context(device):
        for row in rows:
            loss = model(**batch([row], pad, device), use_cache=False).loss
            require(torch.isfinite(loss).item(), "Nonfinite development loss")
            totals[row["kind"]][0] += loss.item() * row["target_tokens"]
            totals[row["kind"]][1] += row["target_tokens"]
    groups = {kind: value[0] / value[1] for kind, value in totals.items()}
    return {"mean_kind_nll": sum(groups.values()) / len(groups), "kind_nll": groups}


def weight_changes(selected, initial):
    result = {}
    for name, parameter in selected.items():
        current = parameter.detach().float().cpu()
        difference = current - initial[name].float()
        result[name] = {"before_sha256": tensor_sha(initial[name]), "after_sha256": tensor_sha(current),
                        "max_abs_delta": difference.abs().max().item(),
                        "l2_delta": difference.double().square().sum().sqrt().item(),
                        "changed_elements": difference.count_nonzero().item(),
                        "effective_bf16_changed_elements": (current.bfloat16() != initial[name].bfloat16()).count_nonzero().item(),
                        "numel": parameter.numel()}
    return result


def read_evaluation(folder, manifest, files):
    rows = []
    for file in files:
        name = Path(file).stem
        rows.extend(dict(r, evaluation_split=name) for r in read_split(folder, manifest, name))
    require(len({r["id"] for r in rows}) == len(rows), "Evaluation IDs overlap")
    return rows


def generate_answers(model, tokenizer, rows, cfg, device, output_file, condition):
    import torch
    from transformers import set_seed
    model.eval()
    records = []
    eos = model.generation_config.eos_token_id
    require(eos is not None, "Model has no explicit generation EOS IDs")
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    with output_file.open("x") as handle, torch.inference_mode(), amp_context(device):
        for row in rows:
            messages = row["messages"][:-1]
            require(row["messages"][-1]["role"] == "assistant", "Evaluation reference must be last")
            ids = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=True, add_generation_prompt=True)
            require(len(ids) <= cfg["max_sequence_tokens"], "Evaluation prompt too long; no silent truncation")
            inputs = torch.tensor([ids], device=device)
            for seed in cfg["eval_seeds"]:
                set_seed(seed)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                        max_new_tokens=cfg["max_new_tokens"], do_sample=True,
                                        temperature=cfg["temperature"], top_p=cfg["top_p"],
                                        top_k=cfg["top_k"], min_p=0.0,
                                        repetition_penalty=cfg["repetition_penalty"],
                                        num_beams=1, eos_token_id=sorted(eos_ids),
                                        pad_token_id=tokenizer.pad_token_id)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated = result[0, len(ids):].tolist()
                raw = tokenizer.decode(generated, skip_special_tokens=False)
                closed = "</think>" in raw
                final = raw.rsplit("</think>", 1)[-1] if closed else None
                if final is not None:
                    for token in tokenizer.all_special_tokens:
                        final = final.replace(token, "")
                    final = final.strip()
                record = {"id": row["id"], "evaluation_split": row["evaluation_split"], "condition": condition,
                          "model_id": cfg["model_id"], "revision": cfg["revision"],
                          "seed": seed, "raw_generated_text": raw, "final_answer": final,
                          "thinking_closed": closed,
                          "output_limit_reached": len(generated) >= cfg["max_new_tokens"] and generated[-1] not in eos_ids,
                          "token_budget_used": len(generated) >= cfg["max_new_tokens"],
                          "ended_with_eos": bool(generated and generated[-1] in eos_ids), "eos_token_ids": sorted(eos_ids),
                          "input_tokens": len(ids), "output_tokens": len(generated), "total_generation_seconds": elapsed,
                          "first_token_seconds": None, "first_final_token_seconds": None,
                          "speech_first_sound_seconds": None, "latency_scope": "complete text generation only; no first-token or audio measurement",
                          "prompt_messages_sha256": fingerprint(messages), "prompt_tokens_sha256": fingerprint(ids),
                          "output_tokens_sha256": fingerprint(generated), "enable_thinking": True,
                          "temperature": cfg["temperature"], "top_p": cfg["top_p"], "top_k": cfg["top_k"],
                          "repetition_penalty": cfg["repetition_penalty"],
                          "max_new_tokens": cfg["max_new_tokens"], "reference": row["messages"][-1]["content"]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(json.dumps({"condition": condition, "id": row["id"], "seed": seed,
                                  "seconds": round(elapsed, 3), "thinking_closed": closed}, ensure_ascii=False), flush=True)
    return {"answers": len(records), "output_limited": sum(r["output_limit_reached"] for r in records),
            "thinking_not_closed": sum(not r["thinking_closed"] for r in records), "sha256": sha(output_file)}


def main():
    global ACTIVE_REPORT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "baseline", "run"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.spec.resolve().parent
    cfg = {"epochs": 2, "learning_rate": 1e-5, "weight_decay": 0.0, "accumulation": 8,
           "seed": 214, "max_sequence_tokens": 2048, "max_new_tokens": 2048,
           "eval_seeds": [214, 215, 216], "temperature": 1.0, "top_p": 0.95, "top_k": 0,
           "repetition_penalty": 1.0, "enable_thinking": True,
           "clip": 1.0, "last_layers": 2, "evaluation_files": ["regression.jsonl", "synthetic-eval.jsonl"]}
    cfg.update(json.loads(args.spec.read_text()))
    require(len(set(cfg["eval_seeds"])) >= 3, "Use at least three fixed distinct evaluation seeds")
    require(cfg["epochs"] >= 1 and cfg["accumulation"] >= 1 and cfg["learning_rate"] > 0, "Invalid training settings")
    require(cfg["enable_thinking"] is True, "This experiment requires native thinking; forcing an empty thinking prompt is excluded")
    require(all(Path(n).name == n and n.endswith(".jsonl") for n in cfg["evaluation_files"]), "Evaluation filenames only")
    snapshot = resolve_path(root, cfg["snapshot_path"])
    folder = resolve_path(root, cfg.get("data_dir", "data"))
    output = args.output or resolve_path(root, cfg.get("output_dir", "results"))
    require(not output.exists(), "Refusing to overwrite an existing run")
    output.mkdir(parents=True, mode=0o700)
    ACTIVE_REPORT_PATH = output / "run.json"
    write_json(ACTIVE_REPORT_PATH, {"status": "preflight", "command": args.command, "config": cfg,
                                  "model_loaded": False, "training_executed": False, "optimizer_steps": 0,
                                  "deployment_changed": False, "quality_verdict": "not_scored"})
    model_files = verify_snapshot(snapshot, cfg["model_id"], cfg["revision"])
    manifest = load_manifest(folder, cfg["evaluation_files"])
    feedback_sha = None
    if cfg.get("feedback_intake_path"):
        feedback_sha = sha(resolve_path(root, cfg["feedback_intake_path"]))
        require(feedback_sha == manifest.get("feedback_intake_sha256"), "Feedback intake is not the frozen fresh snapshot")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    require(tokenizer.pad_token_id is not None, "Tokenizer has no explicit padding token")
    encoded, audit = {}, []
    for split in ("train", "dev"):
        encoded[split] = []
        for row in read_split(folder, manifest, split):
            item = encode_final_answer(tokenizer, row, cfg["max_sequence_tokens"], feedback_sha)
            encoded[split].append(item)
            audit.append({k: v for k, v in item.items() if k not in {"input_ids", "labels"}} |
                         {"split": split, "input_tokens": len(item["input_ids"]),
                          "input_ids_sha256": fingerprint(item["input_ids"]), "labels_sha256": fingerprint(item["labels"])})
    require(encoded["train"] and encoded["dev"], "Training and development data must be nonempty")
    write_json(output / "encoding-audit.json", audit)
    report = {"status": "audit_complete", "command": args.command, "config": cfg,
              "script_sha256": sha(Path(__file__)), "spec_sha256": sha(args.spec),
              "data_manifest_sha256": sha(folder / "manifest.json"),
              "model_file_sha256": model_files, "model_loaded": False, "training_executed": False,
              "optimizer_steps": 0, "deployment_changed": False, "quality_verdict": "not_scored",
              "reasoning_supervision": "none; template-inserted empty reasoning is masked context",
              "limitations": ["Native thinking may change despite absent reasoning supervision.",
                              "Repeated-seed agreement is not independent unseen-user validation.",
                              "Token counts differ across tokenizers; cross-model latency is end-to-end generation only.",
                              "Development NLL selects a candidate, not a deployment decision."],
              "encoded_rows": {k: len(v) for k, v in encoded.items()}, "python": platform.python_version()}
    write_json(output / "run.json", report)
    if args.command == "audit":
        print(json.dumps({"status": "audit_complete", "rows": report["encoded_rows"]}), flush=True)
        return
    import torch
    import transformers
    from safetensors.torch import save_file, load_file
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and re.fullmatch(r"GPU-[A-Za-z0-9-]+", os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "Use the GPU availability launcher; exactly one GPU UUID must be bound")
    require(torch.cuda.is_bf16_supported(), "Selected GPU does not support bf16")
    device = "cuda:0"
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    report.update(status="loading", torch=torch.__version__, transformers=transformers.__version__,
                  gpu_name=torch.cuda.get_device_name(), gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"])
    write_json(output / "run.json", report)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                use_safetensors=True, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(device)
    report.update(model_loaded=True, total_parameters=sum(p.numel() for p in model.parameters()))
    if args.command == "baseline":
        report["selection_closed_before_evaluation"] = True
        rows = read_evaluation(folder, manifest, cfg["evaluation_files"])
        report["baseline_answers"] = generate_answers(model, tokenizer, rows, cfg, device,
                                                       output / "baseline.jsonl", "unmodified_base")
        report["status"] = "baseline_complete"
        write_json(output / "run.json", report)
        return
    require(cfg["model_id"] == "openbmb/MiniCPM5-2B" and model.config.num_hidden_layers == 42,
            "Training recipe was reviewed only for the 42-layer MiniCPM5-2B")
    selected = selected_parameters(model, cfg["last_layers"])
    initial = copy_state(selected)
    frozen_before = frozen_fingerprints(model, selected)
    write_json(output / "frozen-before.json", frozen_before)
    report.update(trainable_parameters=sum(p.numel() for p in selected.values()),
                  trainable_names=list(selected), trainable_dtype="float32", frozen_dtype="bfloat16",
                  strategy="direct_original_projection_weights_in_last_layers", history=[])
    baseline_dev = dev_loss(model, encoded["dev"], tokenizer.pad_token_id, device)
    report["baseline_dev"] = baseline_dev
    optimizer = torch.optim.AdamW(list(selected.values()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best = math.inf
    report["status"] = "training"
    write_json(output / "run.json", report)
    for epoch in range(1, cfg["epochs"] + 1):
        order = list(encoded["train"])
        random.Random(cfg["seed"] + epoch).shuffle(order)
        model.train()
        total_nll, total_tokens = 0.0, 0
        for offset in range(0, len(order), cfg["accumulation"]):
            group = order[offset:offset + cfg["accumulation"]]
            group_tokens = sum(row["target_tokens"] for row in group)
            optimizer.zero_grad(set_to_none=True)
            for row in group:
                with amp_context(device):
                    loss = model(**batch([row], tokenizer.pad_token_id, device), use_cache=False).loss
                require(torch.isfinite(loss).item(), "Nonfinite training loss")
                (loss * row["target_tokens"] / group_tokens).backward()
                total_nll += loss.item() * row["target_tokens"]
                total_tokens += row["target_tokens"]
            torch.nn.utils.clip_grad_norm_(list(selected.values()), cfg["clip"], error_if_nonfinite=True)
            optimizer.step()
            report["optimizer_steps"] += 1
            report["training_executed"] = True
            write_json(output / "run.json", report)
        result = dev_loss(model, encoded["dev"], tokenizer.pad_token_id, device)
        current = copy_state(selected)
        metadata = {"model_id": cfg["model_id"], "revision": cfg["revision"],
                    "epoch": str(epoch), "format": "absolute_selected_original_weights_fp32",
                    "recipe": "last_layers_direct_sft_final_answer_only"}
        save_file(current, str(output / "last_candidate.safetensors"), metadata=metadata)
        if result["mean_kind_nll"] < best:
            best = result["mean_kind_nll"]
            save_file(current, str(output / "candidate.safetensors"), metadata=metadata)
            report["best_trained_epoch"] = epoch
            report["candidate_dev"] = result
        entry = {"epoch": epoch, "train_target_nll": total_nll / total_tokens, "dev": result,
                 "optimizer_steps": report["optimizer_steps"]}
        report["history"].append(entry)
        write_json(output / "run.json", report)
        print(json.dumps(entry), flush=True)
        del current
    report.update(training_executed=True, training_seconds=time.perf_counter() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    frozen_after = frozen_fingerprints(model, selected)
    write_json(output / "frozen-after.json", frozen_after)
    require(frozen_after == frozen_before, "A frozen parameter changed; do not use candidate")
    report["frozen_parameters_unchanged"] = True
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    state = load_file(str(output / "candidate.safetensors"))
    apply_state(selected, state)
    require(all(tensor_sha(selected[n]) == tensor_sha(v) for n, v in state.items()), "Reloaded weights differ")
    report["reloaded_parameter_hashes_match"] = True
    report["weight_changes"] = weight_changes(selected, initial)
    require(any(v["changed_elements"] for v in report["weight_changes"].values()), "No selected weights changed")
    report["effective_bf16_changed_elements"] = sum(v["effective_bf16_changed_elements"] for v in report["weight_changes"].values())
    effective_update = report["effective_bf16_changed_elements"] > 0
    report["candidate_has_effective_bf16_update"] = effective_update
    report["numerical_update_status"] = "effective_bf16_change" if effective_update else "fp32_only_no_effective_bf16_change"
    reloaded_dev = dev_loss(model, encoded["dev"], tokenizer.pad_token_id, device)
    report["reloaded_dev"] = reloaded_dev
    report["reload_nll_delta"] = abs(reloaded_dev["mean_kind_nll"] - best)
    require(report["reload_nll_delta"] <= 1e-5, "Reloaded development loss does not reproduce candidate")
    report.update(candidate_sha256=sha(output / "candidate.safetensors"),
                  candidate_bytes=(output / "candidate.safetensors").stat().st_size,
                  last_candidate_sha256=sha(output / "last_candidate.safetensors"),
                  selected_by_dev_nll="candidate" if effective_update and best < baseline_dev["mean_kind_nll"] else "unmodified_base",
                  selection_closed_before_evaluation=True, status="generating_comparison")
    write_json(output / "run.json", report)
    rows = read_evaluation(folder, manifest, cfg["evaluation_files"])
    apply_state(selected, initial)
    report["baseline_answers"] = generate_answers(model, tokenizer, rows, cfg, device,
                                                   output / "baseline.jsonl", "unmodified_base")
    apply_state(selected, state)
    report["candidate_answers"] = generate_answers(model, tokenizer, rows, cfg, device,
                                                    output / "candidate.jsonl", "direct_layer_candidate")
    report["status"] = "training_and_answers_complete_review_pending"
    write_json(output / "run.json", report)
    print(json.dumps({"status": report["status"], "optimizer_steps": report["optimizer_steps"],
                      "candidate_sha256": report["candidate_sha256"], "selected_by_dev_nll": report["selected_by_dev_nll"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        if ACTIVE_REPORT_PATH is not None and ACTIVE_REPORT_PATH.exists():
            failed = json.loads(ACTIVE_REPORT_PATH.read_text())
            failed.update(failed_phase=failed.get("status"), status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error_type=type(error).__name__, error=str(error)[:2000])
            write_json(ACTIVE_REPORT_PATH, failed)
        raise

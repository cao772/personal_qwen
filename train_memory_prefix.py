#!/usr/bin/env python3
"""Offline-only experimental soft-memory training on a local HF checkpoint.

JSONL rows: {"question": str, "answer": str, "memories": [str, ...]}.
Optional `id`; `timestamp` is required for chronological splitting and must
be ISO 8601 with a timezone. Empty memories are allowed. Every supplied memory
is used. The encoder mean-pools frozen token embeddings and loses word order
and details: this is not Memorilla and not a claim of reliable long-term memory.

Only the memory reader is trained. All reported losses are teacher-forced
answer-token losses, NOT recall accuracy, generated-answer quality, or proof
that personal facts were learned. This script does not download anything.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-declaration", choices=("chronological", "manual"), required=True)
    parser.add_argument("--split-note", default="", help="Required rationale for a manual split")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--reader-dim", type=int, default=128)
    parser.add_argument("--max-question-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--max-memory-doc-tokens", type=int, default=512)
    parser.add_argument("--max-total-memory-tokens", type=int, default=4096)
    parser.add_argument("--max-memories", type=int, default=64)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024,
                        help="Includes soft prefix, prompt, answer and EOS")
    parser.add_argument("--prompt-format", choices=("chat", "plain"), default="chat")
    parser.add_argument("--system-prompt", default=(
        "Answer the user's question using relevant supplied memory. "
        "If the information is not available, say so."
    ))
    args = parser.parse_args()
    positive = ("epochs", "batch_size", "num_slots", "reader_dim", "max_question_tokens",
                "max_prompt_tokens", "max_answer_tokens", "max_memory_doc_tokens",
                "max_total_memory_tokens", "max_memories", "max_sequence_tokens")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        parser.error("--gradient-clip must be finite and positive")
    if args.split_declaration == "manual" and not args.split_note.strip():
        parser.error("a manual split requires --split-note explaining separation")
    return args


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    """Hash large checkpoints without loading their bytes into RAM at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def weight_manifest(model_dir: Path) -> dict[str, dict]:
    files = sorted(path for path in model_dir.rglob("*.safetensors") if path.is_file())
    if not files:
        raise ValueError("local checkpoint has no .safetensors weight files")
    return {
        path.relative_to(model_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
        for path in files
    }


def load_rows(path: Path, max_memories: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{location}: row must be an object")
            for field in ("question", "answer"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{location}: {field} must be nonempty text")
            memories = row.get("memories")
            if not isinstance(memories, list) or any(
                not isinstance(text, str) or not text.strip() for text in memories
            ):
                raise ValueError(f"{location}: memories must be a list of nonempty strings")
            if len(memories) > max_memories:
                raise ValueError(f"{location}: memory count exceeds --max-memories; no truncation")
            row = dict(row)
            row["_location"] = location
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no usable rows")
    return rows


def input_fingerprint(row: dict) -> str:
    normalize = lambda text: " ".join(text.split())
    # Sort only for duplicate detection. Actual memory order is not changed.
    identity = [normalize(row["question"]), sorted(normalize(x) for x in row["memories"])]
    return digest_bytes(json.dumps(identity, ensure_ascii=False).encode("utf-8"))


def timestamp(row: dict) -> datetime:
    raw = row.get("timestamp")
    if not isinstance(raw, str):
        raise ValueError(f"{row['_location']}: chronological split needs timestamp")
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{row['_location']}: invalid ISO 8601 timestamp") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{row['_location']}: timestamp needs an explicit timezone")
    return value.astimezone(timezone.utc)


def check_split(train: list[dict], evaluation: list[dict], declaration: str) -> None:
    fingerprints = []
    record_ids = []
    for name, rows in (("train", train), ("eval", evaluation)):
        keys = [input_fingerprint(row) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate question+memory input inside {name} split")
        fingerprints.append(set(keys))
        ids = [str(row["id"]) for row in rows if row.get("id") is not None]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate id inside {name} split")
        record_ids.append(set(ids))
    if fingerprints[0] & fingerprints[1] or record_ids[0] & record_ids[1]:
        raise ValueError("train/eval overlap detected; use disjoint samples")
    if declaration == "chronological":
        if max(timestamp(row) for row in train) >= min(timestamp(row) for row in evaluation):
            raise ValueError("chronological split requires all train timestamps < all eval timestamps")


def main() -> None:
    args = parse_args()
    args.model_dir = args.model_dir.expanduser().resolve(strict=True)
    if not args.model_dir.is_dir() or not (args.model_dir / "config.json").is_file():
        raise ValueError("--model-dir must be a local HF model directory with config.json")
    source_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(source_config, dict) or source_config.get("model_type") != "qwen3":
        raise ValueError("This training entry supports standard dense Qwen3 only")
    if (args.model_dir / "adapter_config.json").exists():
        raise ValueError("Provide a standalone Qwen3 checkpoint, not an adapter redirect")
    if any(source_config.get(key) for key in ("quantization_config", "compression_config", "num_experts")):
        raise ValueError("Quantized/compressed or MoE checkpoints are outside this baseline")
    args.train_jsonl = args.train_jsonl.expanduser().resolve(strict=True)
    args.eval_jsonl = args.eval_jsonl.expanduser().resolve(strict=True)
    if args.train_jsonl == args.eval_jsonl:
        raise ValueError("train and eval files must be different")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == args.model_dir or args.model_dir in args.output_dir.parents:
        raise ValueError("--output-dir must be outside --model-dir; source checkpoints are not modified")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("--output-dir must be absent or empty; existing results are not overwritten")
    train_rows = load_rows(args.train_jsonl, args.max_memories)
    eval_rows = load_rows(args.eval_jsonl, args.max_memories)
    check_split(train_rows, eval_rows, args.split_declaration)

    # Both offline flags and local_files_only are intentional. No remote code.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from memory_prefix import MemoryPrefixConfig, MemoryPrefixReader, freeze_base_model, prepend_memory_prefix
    except ImportError as exc:
        raise RuntimeError("Training needs locally installed torch and transformers; nothing was installed or downloaded") from exc

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    bf16_available = device_name == "cuda" and torch.cuda.is_bf16_supported()
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if bf16_available else "float32"
    if dtype_name == "bfloat16" and not bf16_available:
        raise ValueError("this baseline permits bfloat16 only on a supporting CUDA device")
    dtype = getattr(torch, dtype_name)
    device = torch.device(device_name)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("local tokenizer must define an EOS token")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.prompt_format == "chat" and not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template; choose --prompt-format plain explicitly")

    def tokenize_rows(rows: list[dict]) -> list[dict]:
        result = []
        for row in rows:
            q_ids = tokenizer.encode(row["question"], add_special_tokens=False)
            memory_ids = [tokenizer.encode(text, add_special_tokens=False) for text in row["memories"]]
            if not q_ids or any(not ids for ids in memory_ids):
                raise ValueError(f"{row['_location']}: tokenizer produced an empty question/memory")
            if len(q_ids) > args.max_question_tokens:
                raise ValueError(f"{row['_location']}: question token limit exceeded; no truncation")
            if any(len(ids) > args.max_memory_doc_tokens for ids in memory_ids):
                raise ValueError(f"{row['_location']}: per-memory token limit exceeded; no truncation")
            if sum(map(len, memory_ids)) > args.max_total_memory_tokens:
                raise ValueError(f"{row['_location']}: total memory token limit exceeded; no truncation")
            if args.prompt_format == "chat":
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": args.system_prompt},
                     {"role": "user", "content": row["question"]}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                )
            else:
                prompt_ids = tokenizer.encode(
                    f"{args.system_prompt}\n\nUser: {row['question']}\nAssistant:",
                    add_special_tokens=True,
                )
            answer_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
            if not answer_ids:
                raise ValueError(f"{row['_location']}: answer produced no tokens")
            if answer_ids[-1] != tokenizer.eos_token_id:
                answer_ids.append(tokenizer.eos_token_id)
            if len(prompt_ids) > args.max_prompt_tokens or len(answer_ids) > args.max_answer_tokens:
                raise ValueError(f"{row['_location']}: prompt/answer token limit exceeded; no truncation")
            if args.num_slots + len(prompt_ids) + len(answer_ids) > args.max_sequence_tokens:
                raise ValueError(f"{row['_location']}: combined sequence token limit exceeded; no truncation")
            result.append({"question_ids": q_ids, "memory_ids": memory_ids,
                           "input_ids": list(prompt_ids) + answer_ids,
                           "labels": [-100] * len(prompt_ids) + answer_ids})
        return result

    train = tokenize_rows(train_rows)
    evaluation = tokenize_rows(eval_rows)
    print("Hashing local .safetensors files to bind the memory reader to this checkpoint.", flush=True)
    base_weight_files = weight_manifest(args.model_dir)
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=dtype,
    ).to(device)
    freeze_base_model(base)
    context_limit = getattr(base.config, "max_position_embeddings", None)
    actual_max = max(len(row["input_ids"]) + args.num_slots for row in train + evaluation)
    if isinstance(context_limit, int) and actual_max > context_limit:
        raise ValueError(f"sequence length {actual_max} exceeds model max_position_embeddings={context_limit}")
    embedding = base.get_input_embeddings()
    hidden_size = embedding.weight.shape[1]
    config = MemoryPrefixConfig(hidden_size, hidden_size, hidden_size, args.reader_dim, args.num_slots)
    reader = MemoryPrefixReader(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def mean_pool(sequences: list[list[int]]):
        if not sequences:
            return torch.zeros((0, hidden_size), device=device, dtype=torch.float32)
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), pad_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, tokens in enumerate(sequences):
            ids[index, :len(tokens)] = torch.tensor(tokens, device=device)
            mask[index, :len(tokens)] = True
        # Only the encoder is detached. Answer tokens never enter this encoder.
        with torch.no_grad():
            encoded = embedding(ids).float()
            pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return pooled.detach()

    def batch_loss(rows: list[dict]):
        batch = len(rows)
        length = max(len(row["input_ids"]) for row in rows)
        ids = torch.full((batch, length), pad_id, device=device, dtype=torch.long)
        mask = torch.zeros((batch, length), device=device, dtype=torch.long)
        labels = torch.full((batch, length), -100, device=device, dtype=torch.long)
        for index, row in enumerate(rows):
            count = len(row["input_ids"])
            ids[index, :count] = torch.tensor(row["input_ids"], device=device)
            mask[index, :count] = 1
            labels[index, :count] = torch.tensor(row["labels"], device=device)
        query_vectors = mean_pool([row["question_ids"] for row in rows])
        max_memories = max(len(row["memory_ids"]) for row in rows)
        memories = torch.zeros((batch, max_memories, hidden_size), device=device, dtype=torch.float32)
        memory_mask = torch.zeros((batch, max_memories), device=device, dtype=torch.bool)
        for index, row in enumerate(rows):
            count = len(row["memory_ids"])
            if count:
                memories[index, :count] = mean_pool(row["memory_ids"])
                memory_mask[index, :count] = True
        prefix = reader(query_vectors, memories, memory_mask)
        with torch.no_grad():
            token_embeddings = embedding(ids)
        inputs = prepend_memory_prefix(token_embeddings, mask, labels, prefix)
        # CRITICAL: do not wrap this decoder forward in no_grad during training.
        # Frozen decoder weights still propagate answer loss to the soft prefix.
        loss = base(**inputs, use_cache=False, return_dict=True).loss
        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError("non-finite/missing loss; training stopped without claiming success")
        target_tokens = int((labels != -100).sum().item())
        return loss, target_tokens

    def evaluate() -> float:
        reader.eval()
        numerator, denominator = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(evaluation), args.batch_size):
                loss, count = batch_loss(evaluation[start:start + args.batch_size])
                numerator += loss.item() * count
                denominator += count
        return numerator / denominator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "training_started", "prototype": "experimental_mean_pool_soft_prefix",
        "not_a_reproduction_of": "Memorilla", "encoder": "frozen_input_embedding_mean_pool",
        "metric_warning": "Teacher-forced answer loss only; no generated-answer or recall evaluation.",
        "reader_config": config.to_dict(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_device": device_name, "resolved_dtype": dtype_name,
        "base_config_sha256": digest_bytes((args.model_dir / "config.json").read_bytes()),
        "tokenizer_files_sha256": {
            name: file_digest(args.model_dir / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                         "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")
            if (args.model_dir / name).is_file()
        },
        "base_weight_files": base_weight_files,
        "base_weight_manifest_sha256": digest_bytes(
            json.dumps(base_weight_files, sort_keys=True).encode("utf-8")
        ),
        "reload_requirement": "Before loading this reader, verify every recorded base weight digest and config digest; reject mismatches.",
        "train_file_sha256": digest_bytes(args.train_jsonl.read_bytes()),
        "eval_file_sha256": digest_bytes(args.eval_jsonl.read_bytes()),
        "train_samples": len(train), "eval_samples": len(evaluation),
        "trainable_parameters": sum(parameter.numel() for parameter in reader.parameters()),
        "base_parameters_frozen": True,
        "split_check_scope": "Input/ID overlap and declared timestamp order; semantic leakage still needs review.",
    }
    config_path = args.output_dir / "memory_config.json"
    config_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    history_path = args.output_dir / "history.jsonl"

    def record(entry: dict) -> None:
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    print("EXPERIMENTAL: training a lossy memory reader only; this is not verified personal memory.", flush=True)
    record({"epoch": 0, "eval_teacher_forced_answer_loss": evaluate(), "reader_state": "random_initialization"})
    shuffle = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        reader.train()
        indices = list(range(len(train)))
        shuffle.shuffle(indices)
        numerator, denominator = 0.0, 0
        for start in range(0, len(indices), args.batch_size):
            rows = [train[index] for index in indices[start:start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss, count = batch_loss(rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), args.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            numerator += loss.item() * count
            denominator += count
        record({"epoch": epoch, "train_teacher_forced_answer_loss": numerator / denominator,
                "eval_teacher_forced_answer_loss": evaluate()})
        torch.save({name: value.detach().cpu() for name, value in reader.state_dict().items()},
                   args.output_dir / "memory_prefix.pt")
    manifest["status"] = "training_completed_not_quality_validated"
    config_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved memory-side weights and loss history to {args.output_dir}. No generation-quality claim.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

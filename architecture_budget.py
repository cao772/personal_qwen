#!/usr/bin/env python3
"""Count standard dense Qwen3 parameters from a config, without weights.

This is an analytical budget, not a loaded-checkpoint measurement or benchmark.
Sources and reference date are recorded in SOURCES.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_parameters(config: dict, intermediate: int | None = None) -> dict:
    if config.get("model_type") != "qwen3":
        raise ValueError("Only standard dense Qwen3 is supported.")
    if config.get("attention_bias", False) or config.get("quantization_config"):
        raise ValueError("Biased attention and quantized checkpoints are not supported.")
    if any(config.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        raise ValueError("MoE is not supported.")
    h = int(config["hidden_size"])
    n = int(config["num_hidden_layers"])
    i = int(intermediate or config["intermediate_size"])
    d = int(config["head_dim"])
    q = int(config["num_attention_heads"])
    kv = int(config["num_key_value_heads"])
    v = int(config["vocab_size"])
    if min(h, n, i, d, q, kv, v) <= 0 or q % kv:
        raise ValueError("Invalid Qwen3 dimensions.")
    embedding = v * h
    lm_head = 0 if config.get("tie_word_embeddings", False) else v * h
    attention = n * (2 * h * q * d + 2 * h * kv * d)
    ffn = n * 3 * h * i
    norms = n * (2 * h + 2 * d) + h
    total = embedding + lm_head + attention + ffn + norms
    return {
        "intermediate_size": i,
        "parameters": total,
        "embedding_parameters": embedding,
        "additional_untied_lm_head_parameters": lm_head,
        "attention_parameters": attention,
        "ffn_parameters": ffn,
        "norm_parameters": norms,
        "raw_bf16_weight_bytes": total * 2,
        "raw_int4_weight_bytes_ideal": total / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("qwen3_1_7b_config_reference.json"))
    parser.add_argument("--intermediate-sizes", type=int, nargs="+", default=[6144, 5376, 4608])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = count_parameters(config)
    candidates = []
    for size in args.intermediate_sizes:
        if not 0 < size <= config["intermediate_size"]:
            raise ValueError("Candidate size must be positive and no larger than the base.")
        candidate = count_parameters(config, size)
        candidate["removed_parameters"] = base["parameters"] - candidate["parameters"]
        candidate["parameter_reduction_percent"] = 100 * candidate["removed_parameters"] / base["parameters"]
        candidate["ffn_channel_reduction_percent"] = 100 * (1 - size / config["intermediate_size"])
        candidates.append(candidate)
    print(json.dumps({
        "status": "analytical_config_budget_not_actual_checkpoint_or_performance",
        "config": args.config.name,
        "assumptions": [
            "Standard dense Qwen3, three bias-free FFN matrices per layer",
            "Q/K RMSNorm parameters shared across heads, matching standard implementation",
            "Only FFN intermediate channels change; all other dimensions fixed",
            "Raw weight bytes exclude tensor headers, quantization scales, caches, activations, optimizer and memory module",
            "Parameter reduction is not a measured latency improvement or capability preservation result"
        ],
        "base": base,
        "candidates": candidates,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

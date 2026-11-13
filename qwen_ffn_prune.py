#!/usr/bin/env python3
"""Conservative, offline FFN channel pruning for dense Hugging Face Qwen3.

Requires only Python and NumPy. No model code is executed and no model is
downloaded. Supports a single, non-quantized model.safetensors (F32/F16/BF16).
The L2 magnitude rule is a demonstration ranking, NOT semantic importance.
Optional --scores-json accepts calibration scores; these are still importance
proxies on a calibration distribution, not proof of separable knowledge.
Structural validity does not establish model quality. Recovery training and
held-out evaluation are still required. ``crop`` is dry-run unless --apply.

Examples:
  python qwen_ffn_prune.py inspect --model-dir /path/to/qwen3
  python qwen_ffn_prune.py plan --model-dir /path/to/qwen3 --keep-ratio 0.75
  python qwen_ffn_prune.py crop --model-dir /path/to/qwen3 \
      --target-intermediate-size 2048 --output-dir /path/to/new-model --apply
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, BinaryIO, Iterator

import numpy as np


class PruningError(ValueError):
    """Input or operation is outside this deliberately narrow tool's scope."""


DTYPES = {"F32": (np.dtype("<f4"), 4),
          "F16": (np.dtype("<f2"), 2),
          "BF16": (np.dtype("<u2"), 2)}
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_SCORES_BYTES = 128 * 1024 * 1024
ASSET_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
    "spiece.model", "generation_config.json", "chat_template.jinja", "LICENSE", "NOTICE",
)
METHOD = "sqrt(L2(gate_row)^2 + L2(up_row)^2 + L2(down_column)^2)"
WARNINGS = [
    "Weight magnitude is only a demonstration ranking; it is not semantic importance.",
    "No guarantee that a removed channel is unused, or that model quality is preserved.",
    "Recovery training and held-out evaluation have not been performed.",
    "Validation covers config, FFN shapes and safetensors integrity, not a real HF forward pass.",
    "Only a single non-quantized dense Qwen3 model.safetensors is supported.",
]


def _json_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PruningError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        result = json.loads(raw, object_pairs_hook=_json_unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PruningError(f"Invalid JSON in {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise PruningError(f"{label} must contain a JSON object")
    return result


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise PruningError(f"config.{key} must be a positive integer")
    return value


def _snapshot(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def _file_sha256(path: Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class SafeTensorFile:
    """Read and validate a safetensors header without loading tensor payloads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise PruningError(f"Missing safetensors file: {self.path}")
        self.snapshot = _snapshot(self.path)
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise PruningError("Truncated safetensors length prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= MAX_HEADER_BYTES:
                raise PruningError("Invalid or excessive safetensors header size")
            if 8 + header_size > self.snapshot["size"]:
                raise PruningError("Truncated safetensors header")
            raw_header = stream.read(header_size)
        if not raw_header.startswith(b"{"):
            raise PruningError("Safetensors header must begin with '{'")
        header = _parse_json(raw_header, "safetensors header")
        self.header_sha256 = hashlib.sha256(prefix + raw_header).hexdigest()
        self.data_start = 8 + header_size
        self.metadata = header.pop("__metadata__", {})
        if not isinstance(self.metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in self.metadata.items()
        ):
            raise PruningError("Safetensors metadata must be string-to-string")
        if self.metadata.get("format") not in (None, "pt"):
            raise PruningError("Only PyTorch-format HF safetensors metadata is supported")
        self.tensors: dict[str, TensorInfo] = {}
        for name, item in header.items():
            if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
                raise PruningError(f"Invalid tensor header: {name}")
            dtype = item["dtype"]
            if not isinstance(dtype, str) or dtype not in DTYPES:
                raise PruningError(f"Unsupported dtype {dtype!r} in {name}; use F32/F16/BF16")
            shape, offsets = item["shape"], item["data_offsets"]
            if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                raise PruningError(f"Invalid shape for {name}")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(x) is not int or x < 0 for x in offsets)
                    or offsets[1] < offsets[0]):
                raise PruningError(f"Invalid offsets for {name}")
            tensor = TensorInfo(name, dtype, tuple(shape), *offsets)
            if tensor.nbytes != tensor.elements * DTYPES[dtype][1]:
                raise PruningError(f"Shape/byte-size mismatch for {name}")
            self.tensors[name] = tensor
        if not self.tensors:
            raise PruningError("Safetensors file has no tensors")
        cursor = 0
        for tensor in self.ordered_tensors():
            if tensor.start != cursor:
                raise PruningError("Safetensors offsets overlap or contain gaps")
            cursor = tensor.end
        if self.data_start + cursor != self.snapshot["size"]:
            raise PruningError("Safetensors payload is truncated or has trailing bytes")

    def ordered_tensors(self) -> list[TensorInfo]:
        # Empty tensors at a boundary precede a nonempty tensor at that boundary.
        return sorted(self.tensors.values(), key=lambda t: (t.start, t.end, t.name))

    def array(self, name: str) -> np.memmap:
        tensor = self.tensors[name]
        if not tensor.elements:
            raise PruningError("An empty tensor cannot be memory-mapped")
        return np.memmap(self.path, mode="r", dtype=DTYPES[tensor.dtype][0],
                         offset=self.data_start + tensor.start, shape=tensor.shape,
                         order="C")


@dataclass
class ModelSpec:
    directory: Path
    config: dict[str, Any]
    config_sha256: str
    weights: SafeTensorFile
    hidden_size: int
    intermediate_size: int
    layers: int


def load_model_spec(model_dir: str | Path) -> ModelSpec:
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise PruningError(f"Model directory does not exist: {directory}")
    if ((directory / "model.safetensors.index.json").exists()
            or any(p.name != "model.safetensors" for p in directory.glob("*.safetensors"))):
        raise PruningError("Sharded/extra safetensors are unsupported; require only model.safetensors")
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise PruningError("Missing Hugging Face config.json")
    raw_config = config_path.read_bytes()
    config = _parse_json(raw_config, "config.json")
    if config.get("model_type") != "qwen3":
        raise PruningError("Only model_type='qwen3' dense models are supported")
    architectures = config.get("architectures")
    if architectures is not None and architectures != ["Qwen3ForCausalLM"]:
        raise PruningError("Only the standard Qwen3ForCausalLM architecture is supported")
    if config.get("auto_map"):
        raise PruningError("Custom remote-code models are unsupported")
    for key in ("quantization_config", "compression_config", "quantization_method"):
        if config.get(key) is not None:
            raise PruningError(f"Quantized/compressed config is unsupported: {key}")
    for key in ("num_experts", "num_local_experts", "num_experts_per_tok", "moe_intermediate_size"):
        if config.get(key) not in (None, 0):
            raise PruningError(f"MoE config is unsupported: {key}")
    hidden = _positive_int(config, "hidden_size")
    intermediate = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    _positive_int(config, "num_attention_heads")
    if "num_key_value_heads" in config:
        _positive_int(config, "num_key_value_heads")
    weights = SafeTensorFile(directory / "model.safetensors")
    expected_mlp = {}
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp."
        expected_mlp.update({prefix + "gate_proj.weight": (intermediate, hidden),
                             prefix + "up_proj.weight": (intermediate, hidden),
                             prefix + "down_proj.weight": (hidden, intermediate)})
    for name, shape in expected_mlp.items():
        tensor = weights.tensors.get(name)
        if tensor is None or tensor.shape != shape:
            actual = None if tensor is None else tensor.shape
            raise PruningError(f"Missing or wrong FFN shape: {name}: expected {shape}, got {actual}")
    for name in weights.tensors:
        if ".mlp." in name and name not in expected_mlp:
            raise PruningError(f"Unsupported MLP tensor/bias/expert: {name}")
        if re.search(r"(?:^|\.)(?:qweight|qzeros|scales|zeros|g_idx|weight_scale)(?:\.|$)", name):
            raise PruningError(f"Quantization tensor is unsupported: {name}")
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match and int(match.group(1)) >= layers:
            raise PruningError(f"Layer index exceeds config: {name}")
    return ModelSpec(directory, config, hashlib.sha256(raw_config).hexdigest(),
                     weights, hidden, intermediate, layers)


def _source_summary(spec: ModelSpec) -> dict[str, Any]:
    return {"model_directory": str(spec.directory),
            "config_sha256": spec.config_sha256,
            "safetensors_header_sha256": spec.weights.header_sha256,
            "safetensors_stat": spec.weights.snapshot,
            "fingerprint_note": "Header digest and file stat, not a full weight-payload digest."}


def inspect_model(model_dir: str | Path) -> dict[str, Any]:
    spec = load_model_spec(model_dir)
    return {"model_type": "qwen3", "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size, "num_hidden_layers": spec.layers,
            "num_attention_heads": spec.config["num_attention_heads"],
            "num_key_value_heads": spec.config.get("num_key_value_heads"),
            "tensor_count": len(spec.weights.tensors),
            "stored_tensor_elements": sum(t.elements for t in spec.weights.tensors.values()),
            "tensor_payload_bytes": sum(t.nbytes for t in spec.weights.tensors.values()),
            "dtypes": sorted({t.dtype for t in spec.weights.tensors.values()}),
            "source": _source_summary(spec), "warnings": WARNINGS}


def _chunk_bytes(chunk_mib: int) -> int:
    if type(chunk_mib) is not int or not 1 <= chunk_mib <= 1024:
        raise PruningError("chunk_mib must be an integer between 1 and 1024")
    return chunk_mib * 1024 * 1024


def _float32(block: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        # BF16 is kept as raw uint16 during copying; expansion is for scoring only.
        bits = np.asarray(block, dtype=np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    return np.asarray(block, dtype=np.float32)


def score_channels(spec: ModelSpec, layer: int, chunk_mib: int = 16) -> np.ndarray:
    budget = _chunk_bytes(chunk_mib)
    result = np.zeros(spec.intermediate_size, dtype=np.float64)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.{layer}.mlp.{projection}.weight"
        tensor = spec.weights.tensors[name]
        array = spec.weights.array(name)
        # At least one row is processed. Its width is the minimum block footprint.
        rows_per = max(1, budget // (tensor.shape[1] * 8))
        for start in range(0, tensor.shape[0], rows_per):
            stop = min(start + rows_per, tensor.shape[0])
            block = _float32(array[start:stop], tensor.dtype)
            if not np.isfinite(block).all():
                raise PruningError(f"Non-finite FFN weights cannot be ranked: {name}")
            if projection == "down_proj":
                result += np.einsum("ij,ij->j", block, block, dtype=np.float64,
                                    optimize=False)
            else:
                result[start:stop] += np.einsum("ij,ij->i", block, block,
                                               dtype=np.float64, optimize=False)
        del block, array
    return np.sqrt(result)


def _target_size(original: int, keep_ratio: float | None,
                 target_intermediate_size: int | None) -> int:
    if keep_ratio is not None and target_intermediate_size is not None:
        raise PruningError("Specify either keep_ratio or target_intermediate_size")
    if target_intermediate_size is not None:
        if type(target_intermediate_size) is not int:
            raise PruningError("target_intermediate_size must be an integer")
        target = target_intermediate_size
    else:
        ratio = 0.75 if keep_ratio is None else keep_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio < 1:
            raise PruningError("keep_ratio must be finite and strictly between 0 and 1")
        target = math.floor(original * ratio)
    if not 1 <= target < original:
        raise PruningError(f"Target intermediate size must be in [1, {original - 1}]")
    return target


def _check_source_unchanged(spec: ModelSpec) -> None:
    if _snapshot(spec.weights.path) != spec.weights.snapshot:
        raise PruningError("Source weight file changed during this operation")
    if hashlib.sha256((spec.directory / "config.json").read_bytes()).hexdigest() != spec.config_sha256:
        raise PruningError("Source config changed during this operation")


def _read_calibration_scores(spec: ModelSpec, scores_json: str | Path) -> tuple[dict, dict]:
    path = Path(scores_json).expanduser().resolve()
    if not path.is_file() or path.stat().st_size > MAX_SCORES_BYTES:
        raise PruningError("Calibration scores must be a JSON file no larger than 128 MiB")
    raw = path.read_bytes()
    data = _parse_json(raw, "calibration scores")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise PruningError("Calibration schema_version must be 1")
    if data.get("score_method") != "activation_weight_energy":
        raise PruningError("Supported calibration score_method is activation_weight_energy")
    if data.get("model_config_sha256") != spec.config_sha256:
        raise PruningError("Calibration model_config_sha256 does not match source config.json")
    weights_digest = data.get("model_weights_sha256")
    if not isinstance(weights_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", weights_digest):
        raise PruningError("Calibration model_weights_sha256 must identify the complete source checkpoint")
    expected_dimensions = {"num_hidden_layers": spec.layers, "hidden_size": spec.hidden_size,
                           "intermediate_size": spec.intermediate_size}
    dimensions = data.get("model_dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(expected_dimensions) or any(
        type(dimensions.get(key)) is not int or dimensions[key] != value
        for key, value in expected_dimensions.items()
    ):
        raise PruningError("Calibration model_dimensions do not match the source model")
    layers = data.get("layer_scores")
    if not isinstance(layers, dict) or set(layers) != {str(i) for i in range(spec.layers)}:
        raise PruningError("Calibration layer_scores must contain exactly every model layer")
    arrays = {}
    for layer, values in layers.items():
        if (not isinstance(values, list) or len(values) != spec.intermediate_size
                or any(type(value) not in (int, float) for value in values)):
            raise PruningError(f"Calibration scores for layer {layer} have invalid length/type")
        try:
            scores = np.asarray(values, dtype=np.float64)
        except (OverflowError, ValueError) as exc:
            raise PruningError(f"Calibration scores for layer {layer} cannot be represented") from exc
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise PruningError(f"Calibration scores for layer {layer} must be finite and nonnegative")
        arrays[layer] = scores
    # Same architecture/config does not identify a checkpoint. Stream the complete
    # source once so scores from a different fine-tune cannot silently be reused.
    if _file_sha256(spec.weights.path) != weights_digest:
        raise PruningError("Calibration model_weights_sha256 does not match source model.safetensors")
    _check_source_unchanged(spec)
    provenance = {"path": str(path), "scores_file_sha256": hashlib.sha256(raw).hexdigest(),
                  "schema_version": 1, "score_method": data["score_method"],
                  "model_config_sha256": data["model_config_sha256"],
                  "model_weights_sha256": weights_digest,
                  "model_dimensions": dimensions,
                  "calibration_data_sha256": data.get("calibration_data_sha256"),
                  "calibration": data.get("calibration"),
                  "binding_note": "Configuration, dimensions and the complete source model.safetensors SHA256 are checked."}
    return arrays, provenance


def _make_plan(spec: ModelSpec, keep_ratio: float | None,
               target_intermediate_size: int | None, chunk_mib: int,
               scores_json: str | Path | None = None) -> dict[str, Any]:
    _chunk_bytes(chunk_mib)
    target = _target_size(spec.intermediate_size, keep_ratio, target_intermediate_size)
    external_scores, score_provenance = ({}, None) if scores_json is None else _read_calibration_scores(spec, scores_json)
    layer_plans = []
    for layer in range(spec.layers):
        scores = external_scores[str(layer)] if scores_json is not None else score_channels(spec, layer, chunk_mib)
        # Index is a deterministic tie-breaker; original channel order is retained.
        ranking = np.lexsort((np.arange(len(scores)), -scores))
        kept = np.sort(ranking[:target]).tolist()
        layer_plans.append({"layer": layer, "kept_channel_indices": kept,
                            "original_channels": spec.intermediate_size,
                            "retained_channels": target,
                            "score_min": float(scores.min()), "score_max": float(scores.max())})
    _check_source_unchanged(spec)
    before = sum(t.elements for t in spec.weights.tensors.values())
    removed = 3 * spec.hidden_size * (spec.intermediate_size - target) * spec.layers
    removed_bytes = 0
    changes = {}
    for layer in range(spec.layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.{projection}.weight"
            tensor = spec.weights.tensors[name]
            shape = [spec.hidden_size, target] if projection == "down_proj" else [target, spec.hidden_size]
            removed_bytes += (tensor.elements - math.prod(shape)) * DTYPES[tensor.dtype][1]
            changes[name] = {"original_shape": list(tensor.shape), "new_shape": shape,
                             "dtype": tensor.dtype}
    warnings = list(WARNINGS)
    if scores_json is not None:
        warnings[0] = "Activation-weight energy is a calibration-distribution proxy, not proof of semantic separability."
    return {"format_version": 1, "operation": "structured_ffn_channel_crop",
            "source": _source_summary(spec), "model_type": "qwen3",
            "selection_method": METHOD if scores_json is None else "activation_weight_energy",
            "score_method": "weight_magnitude_l2" if scores_json is None else "activation_weight_energy",
            "calibration_score_source": score_provenance,
            "semantic_importance_estimated": False,
            "selection_note": ("Demonstration weight-magnitude proxy, without calibration data." if scores_json is None
                               else "Calibration activation/weight proxy; it does not identify removable topics or guarantee retained quality."),
            "original_intermediate_size": spec.intermediate_size,
            "target_intermediate_size": target, "num_hidden_layers": spec.layers,
            "stored_tensor_elements_before": before,
            "stored_tensor_elements_after": before - removed,
            "removed_ffn_weight_parameters": removed,
            "removed_stored_elements_fraction": removed / before,
            "removed_tensor_payload_bytes": removed_bytes,
            "count_note": "Counts stored tensor scalar elements; shared/tied tensors are counted as stored.",
            "unchanged_config_fields": {k: spec.config.get(k) for k in (
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings")},
            "layers": layer_plans, "changed_tensors": changes,
            "recovery_training": "not_performed_required_before_quality_claims",
            "real_model_inference_validation": "not_performed",
            "warnings": warnings, "dry_run": True}


def plan_pruning(model_dir: str | Path, *, keep_ratio: float | None = None,
                 target_intermediate_size: int | None = None,
                 chunk_mib: int = 16, scores_json: str | Path | None = None) -> dict[str, Any]:
    return _make_plan(load_model_spec(model_dir), keep_ratio,
                       target_intermediate_size, chunk_mib, scores_json)


class _HashWriter:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.hash = hashlib.sha256()

    def write(self, data: bytes | memoryview) -> None:
        if self.stream.write(data) != len(data):
            raise OSError("Short write")
        self.hash.update(data)


def _copy_range(source: BinaryIO, writer: _HashWriter, offset: int,
                 size: int, chunk_bytes: int) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        data = source.read(min(remaining, chunk_bytes))
        if not data:
            raise PruningError("Source payload became truncated during copying")
        writer.write(data)
        remaining -= len(data)


def _runs(indices: list[int]) -> Iterator[tuple[int, int]]:
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            yield start, previous + 1
            start = index
        previous = index
    yield start, previous + 1


def _write_weights(spec: ModelSpec, plan: dict[str, Any], destination: Path,
                    chunk_mib: int) -> str:
    budget = _chunk_bytes(chunk_mib)
    header: dict[str, Any] = {"__metadata__": dict(spec.weights.metadata)}
    header["__metadata__"].setdefault("format", "pt")
    header["__metadata__"]["ffn_pruning"] = plan["score_method"] + "_requires_recovery_training"
    cursor = 0
    for tensor in spec.weights.ordered_tensors():
        change = plan["changed_tensors"].get(tensor.name)
        shape = change["new_shape"] if change else list(tensor.shape)
        size = math.prod(shape) * DTYPES[tensor.dtype][1]
        header[tensor.name] = {"dtype": tensor.dtype, "shape": shape,
                               "data_offsets": [cursor, cursor + size]}
        cursor += size
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > MAX_HEADER_BYTES:
        raise PruningError("Output safetensors header exceeds supported size")
    with spec.weights.path.open("rb") as source, destination.open("xb") as output:
        writer = _HashWriter(output)
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        for tensor in spec.weights.ordered_tensors():
            if tensor.name not in plan["changed_tensors"]:
                _copy_range(source, writer, spec.weights.data_start + tensor.start,
                            tensor.nbytes, budget)
                continue
            layer = int(tensor.name.split(".")[2])
            indices = plan["layers"][layer]["kept_channel_indices"]
            itemsize = DTYPES[tensor.dtype][1]
            if ".down_proj." not in tensor.name:
                rowbytes = tensor.shape[1] * itemsize
                for start, stop in _runs(indices):
                    _copy_range(source, writer,
                                spec.weights.data_start + tensor.start + start * rowbytes,
                                (stop - start) * rowbytes, budget)
            else:
                array = spec.weights.array(tensor.name)
                rows_per = max(1, budget // (tensor.shape[1] * itemsize * 3))
                for start in range(0, tensor.shape[0], rows_per):
                    block = np.ascontiguousarray(array[start:start + rows_per, indices])
                    writer.write(memoryview(block).cast("B"))
                del block, array
        output.flush()
        os.fsync(output.fileno())
        digest = writer.hash.hexdigest()
    return digest


def _new_destination(model_directory: Path, output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().absolute()
    if os.path.lexists(output):
        raise PruningError(f"Output must be a completely new directory: {output}")
    output = output.resolve()
    if output == model_directory or model_directory in output.parents:
        raise PruningError("Output cannot be inside the source model directory")
    if not output.parent.is_dir():
        raise PruningError("Output parent directory must already exist")
    return output


def crop_model(model_dir: str | Path, output_dir: str | Path, *,
                keep_ratio: float | None = None,
                target_intermediate_size: int | None = None,
                apply: bool = False, chunk_mib: int = 16,
                scores_json: str | Path | None = None) -> dict[str, Any]:
    """Plan by default; write only with apply=True, always to a new directory."""
    spec = load_model_spec(model_dir)
    output = _new_destination(spec.directory, output_dir)
    plan = _make_plan(spec, keep_ratio, target_intermediate_size, chunk_mib, scores_json)
    plan["output_directory"] = str(output)
    if not apply:
        return plan
    _check_source_unchanged(spec)
    # Exclusive mkdir prevents overwriting an existing model or another run.
    output.mkdir(mode=0o700)
    try:
        digest = _write_weights(spec, plan, output / "model.safetensors", chunk_mib)
        config = copy.deepcopy(spec.config)
        config["intermediate_size"] = plan["target_intermediate_size"]
        (output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied = []
        for filename in ASSET_FILES:
            source = spec.directory / filename
            if source.is_file():
                shutil.copyfile(source, output / filename)
                copied.append(filename)
        _check_source_unchanged(spec)
        load_model_spec(output)  # Structural check of the actual bytes just written.
        manifest = copy.deepcopy(plan)
        manifest.update({"dry_run": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                         "output_safetensors_sha256": digest, "copied_assets": copied,
                         "tensor_shapes": {
                             name: {"original_shape": list(tensor.shape),
                                    "new_shape": plan["changed_tensors"].get(name, {}).get(
                                        "new_shape", list(tensor.shape)),
                                    "dtype": tensor.dtype,
                                    "changed": name in plan["changed_tensors"]}
                             for name, tensor in spec.weights.tensors.items()}})
        (output / "pruning_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    except BaseException:
        # This directory was created exclusively by this call; source is read-only.
        shutil.rmtree(output)
        raise


def _cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in report.items()
              if key not in {"layers", "changed_tensors", "tensor_shapes"}}
    if "layers" in report:
        result["layer_selection_summary"] = [
            {key: value for key, value in layer.items() if key != "kept_channel_indices"}
            for layer in report["layers"]]
    if report.get("dry_run"):
        result["write_status"] = "No files written. crop requires --apply to create a model."
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "plan", "crop"):
        sub = commands.add_parser(name)
        sub.add_argument("--model-dir", required=True, type=Path)
        if name != "inspect":
            target = sub.add_mutually_exclusive_group()
            target.add_argument("--keep-ratio", type=float, help="Retained FFN fraction; default 0.75")
            target.add_argument("--target-intermediate-size", type=int)
            sub.add_argument("--chunk-mib", type=int, default=16,
                             help="Block-work budget (plus memmaps and channel vectors); default 16")
            sub.add_argument("--scores-json", type=Path,
                             help="Optional activation_weight_energy calibration JSON; otherwise magnitude demo")
        if name == "crop":
            sub.add_argument("--output-dir", required=True, type=Path)
            sub.add_argument("--apply", action="store_true",
                             help="Actually write a new checkpoint; without this flag only plan")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            report = inspect_model(args.model_dir)
        else:
            options = {"keep_ratio": args.keep_ratio,
                       "target_intermediate_size": args.target_intermediate_size,
                       "chunk_mib": args.chunk_mib, "scores_json": args.scores_json}
            if args.command == "plan":
                report = plan_pruning(args.model_dir, **options)
            else:
                report = crop_model(args.model_dir, args.output_dir, apply=args.apply, **options)
        print(json.dumps(_cli_summary(report), ensure_ascii=False, indent=2))
        return 0
    except (PruningError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

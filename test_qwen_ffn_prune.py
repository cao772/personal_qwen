"""Synthetic structural tests only. These do not run or evaluate a real Qwen model."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from qwen_ffn_prune import (
    PruningError, SafeTensorFile, crop_model, inspect_model, load_model_spec,
    main, plan_pruning,
)


def write_synthetic_safetensors(path, arrays, dtype="F32"):
    """Independent tiny fixture writer; no dependency on production writer."""
    header = {"__metadata__": {"format": "pt", "fixture": "synthetic_only"}}
    payloads = []
    cursor = 0
    for name, values in arrays.items():
        if dtype == "BF16":
            # The fixture uses truncation; the tested crop must preserve these bits.
            data = (np.asarray(values, dtype="<f4").view("<u4") >> 16).astype("<u2")
        else:
            data = np.asarray(values, dtype={"F32": "<f4", "F16": "<f2", "I32": "<i4"}[dtype])
        raw = data.tobytes(order="C")
        header[name] = {"dtype": dtype, "shape": list(data.shape),
                        "data_offsets": [cursor, cursor + len(raw)]}
        payloads.append(raw)
        cursor += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def make_fixture(directory: Path, dtype="F32"):
    directory.mkdir()
    config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
              "hidden_size": 3, "intermediate_size": 6, "num_hidden_layers": 2,
              "num_attention_heads": 1, "num_key_value_heads": 1,
              "head_dim": 3, "vocab_size": 7, "tie_word_embeddings": False,
              "torch_dtype": {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}.get(dtype, "int32")}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    rng = np.random.default_rng(733)
    arrays = {
        "model.embed_tokens.weight": rng.normal(size=(7, 3)).astype(np.float32),
        "model.norm.weight": np.ones(3, dtype=np.float32),
        "lm_head.weight": rng.normal(size=(7, 3)).astype(np.float32),
    }
    # The fixture deliberately includes non-FFN tensors to check exact preservation.
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        arrays[prefix + "self_attn.q_proj.weight"] = rng.normal(size=(3, 3)).astype(np.float32)
        arrays[prefix + "input_layernorm.weight"] = np.arange(1, 4, dtype=np.float32)
        scale = np.array([0.05, 1.2, 0.1, 2.0, 0.2, 1.0], dtype=np.float32)
        arrays[prefix + "mlp.gate_proj.weight"] = rng.normal(size=(6, 3)).astype(np.float32) * scale[:, None]
        arrays[prefix + "mlp.up_proj.weight"] = rng.normal(size=(6, 3)).astype(np.float32) * scale[:, None]
        arrays[prefix + "mlp.down_proj.weight"] = rng.normal(size=(3, 6)).astype(np.float32) * scale[None, :]
    write_synthetic_safetensors(directory / "model.safetensors", arrays, dtype)
    (directory / "tokenizer.json").write_text('{"synthetic_tokenizer": true}\n')
    (directory / "tokenizer_config.json").write_text('{"synthetic": true}\n')
    (directory / "generation_config.json").write_text('{"max_new_tokens": 8}\n')
    (directory / "do_not_copy.py").write_text("raise RuntimeError('must not be copied or executed')\n")
    return config, arrays


def snapshot_files(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def raw_tensor(reader, name):
    tensor = reader.tensors[name]
    with reader.path.open("rb") as stream:
        stream.seek(reader.data_start + tensor.start)
        return stream.read(tensor.nbytes)


def as_float(reader, name):
    tensor = reader.tensors[name]
    array = np.array(reader.array(name), copy=True)
    if tensor.dtype == "BF16":
        return (array.astype(np.uint32) << 16).view(np.float32)
    return array.astype(np.float32)


class PruningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.config, self.arrays = make_fixture(self.source)

    def test_inspect_and_plan_counts_are_actual_stored_elements(self):
        inspected = inspect_model(self.source)
        expected = sum(a.size for a in self.arrays.values())
        self.assertEqual(inspected["stored_tensor_elements"], expected)
        plan = plan_pruning(self.source, target_intermediate_size=3, chunk_mib=1)
        removed = 2 * 3 * 3 * 3
        self.assertEqual(plan["removed_ffn_weight_parameters"], removed)
        self.assertEqual(plan["stored_tensor_elements_after"], expected - removed)
        self.assertEqual(plan["removed_tensor_payload_bytes"], removed * 4)
        self.assertFalse(plan["semantic_importance_estimated"])
        for layer in plan["layers"]:
            self.assertEqual(len(layer["kept_channel_indices"]), 3)
            self.assertEqual(layer["kept_channel_indices"], sorted(layer["kept_channel_indices"]))

    def test_default_crop_and_cli_are_dry_run(self):
        before = snapshot_files(self.source)
        output = self.root / "dry-run"
        plan = crop_model(self.source, output, target_intermediate_size=3)
        self.assertTrue(plan["dry_run"])
        self.assertFalse(output.exists())
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            result = main(["crop", "--model-dir", str(self.source), "--output-dir", str(output),
                           "--keep-ratio", "0.5"])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(printed.getvalue())["dry_run"])
        self.assertFalse(output.exists())
        self.assertEqual(before, snapshot_files(self.source))

    def test_three_projections_crop_together_config_and_source_preserved(self):
        before = snapshot_files(self.source)
        output = self.root / "cropped"
        manifest = crop_model(self.source, output, target_intermediate_size=3, apply=True, chunk_mib=1)
        self.assertFalse(manifest["dry_run"])
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config, dict(self.config, intermediate_size=3))
        self.assertEqual(snapshot_files(self.source), before)
        self.assertFalse((output / "do_not_copy.py").exists())
        self.assertEqual((output / "tokenizer.json").read_bytes(), (self.source / "tokenizer.json").read_bytes())
        original, cropped = SafeTensorFile(self.source / "model.safetensors"), SafeTensorFile(output / "model.safetensors")
        for name in original.tensors:
            if ".mlp." not in name:
                self.assertEqual(original.tensors[name].shape, cropped.tensors[name].shape)
                self.assertEqual(raw_tensor(original, name), raw_tensor(cropped, name))
        for layer in range(2):
            keep = manifest["layers"][layer]["kept_channel_indices"]
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.{projection}.weight"
                array = as_float(original, name)
                expected = array[:, keep] if projection == "down_proj" else array[keep, :]
                np.testing.assert_array_equal(as_float(cropped, name), expected)
        self.assertEqual(manifest["output_safetensors_sha256"],
                         hashlib.sha256((output / "model.safetensors").read_bytes()).hexdigest())
        saved = json.loads((output / "pruning_manifest.json").read_text())
        self.assertEqual(saved["tensor_shapes"]["model.embed_tokens.weight"]["changed"], False)
        self.assertEqual(saved["recovery_training"], "not_performed_required_before_quality_claims")

    def test_cropped_ffn_equals_original_with_removed_channels_masked(self):
        output = self.root / "cropped"
        manifest = crop_model(self.source, output, target_intermediate_size=3, apply=True)
        original, cropped = SafeTensorFile(self.source / "model.safetensors"), SafeTensorFile(output / "model.safetensors")
        inputs = np.random.default_rng(66).normal(size=(5, 3))
        for layer in range(2):
            prefix = f"model.layers.{layer}.mlp."
            gate = as_float(original, prefix + "gate_proj.weight").astype(np.float64)
            up = as_float(original, prefix + "up_proj.weight").astype(np.float64)
            down = as_float(original, prefix + "down_proj.weight").astype(np.float64)
            hidden = inputs @ gate.T
            hidden = hidden / (1.0 + np.exp(-hidden)) * (inputs @ up.T)
            mask = np.zeros(6)
            mask[manifest["layers"][layer]["kept_channel_indices"]] = 1.0
            reference = (hidden * mask) @ down.T
            gate2 = as_float(cropped, prefix + "gate_proj.weight").astype(np.float64)
            up2 = as_float(cropped, prefix + "up_proj.weight").astype(np.float64)
            down2 = as_float(cropped, prefix + "down_proj.weight").astype(np.float64)
            h2 = inputs @ gate2.T
            result = (h2 / (1.0 + np.exp(-h2)) * (inputs @ up2.T)) @ down2.T
            np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)

    def test_f16_and_bf16_preserve_raw_retained_values(self):
        for dtype in ("F16", "BF16"):
            with self.subTest(dtype=dtype):
                source = self.root / dtype
                make_fixture(source, dtype)
                output = self.root / (dtype + "-out")
                manifest = crop_model(source, output, target_intermediate_size=3, apply=True)
                original = SafeTensorFile(source / "model.safetensors")
                cropped = SafeTensorFile(output / "model.safetensors")
                for name, tensor in original.tensors.items():
                    self.assertEqual(cropped.tensors[name].dtype, dtype)
                    if ".mlp." not in name:
                        self.assertEqual(raw_tensor(original, name), raw_tensor(cropped, name))
                    else:
                        layer = int(name.split(".")[2])
                        keep = manifest["layers"][layer]["kept_channel_indices"]
                        original_array = np.array(original.array(name), copy=True)
                        expected = original_array[:, keep] if ".down_proj." in name else original_array[keep, :]
                        self.assertEqual(raw_tensor(cropped, name), expected.tobytes(order="C"))

    def test_ties_have_stable_channel_order(self):
        arrays = {name: np.ones_like(array) for name, array in self.arrays.items()}
        write_synthetic_safetensors(self.source / "model.safetensors", arrays)
        plan = plan_pruning(self.source, target_intermediate_size=3)
        self.assertEqual([x["kept_channel_indices"] for x in plan["layers"]], [[0, 1, 2], [0, 1, 2]])

    def test_output_must_be_new_and_outside_source(self):
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("do not modify")
        for output in (existing, self.source, self.source / "new-child"):
            with self.subTest(output=str(output)), self.assertRaises(PruningError):
                crop_model(self.source, output, target_intermediate_size=3, apply=True)
        self.assertEqual(marker.read_text(), "do not modify")
        self.assertFalse((self.source / "new-child").exists())

    def test_unsupported_config_and_shards_fail_fast(self):
        path = self.source / "config.json"
        cases = [dict(self.config, model_type="qwen2"),
                 dict(self.config, model_type="qwen3_moe"),
                 dict(self.config, num_experts=8),
                 dict(self.config, quantization_config={"bits": 4}),
                 dict(self.config, architectures=["CustomForCausalLM"]),
                 dict(self.config, auto_map={"AutoModel": "custom.Model"})]
        for config in cases:
            with self.subTest(config=config):
                path.write_text(json.dumps(config))
                output = self.root / "unsupported-out"
                with self.assertRaises(PruningError):
                    crop_model(self.source, output, target_intermediate_size=3, apply=True)
                self.assertFalse(output.exists())
        path.write_text(json.dumps(self.config))
        (self.source / "model.safetensors.index.json").write_text("{}")
        with self.assertRaisesRegex(PruningError, "Sharded"):
            load_model_spec(self.source)

    def test_unsupported_dtype_bias_and_shape_rejected(self):
        write_synthetic_safetensors(self.source / "model.safetensors", self.arrays, "I32")
        with self.assertRaisesRegex(PruningError, "Unsupported dtype"):
            load_model_spec(self.source)
        bad = dict(self.arrays)
        bad["model.layers.0.mlp.gate_proj.bias"] = np.zeros(6)
        write_synthetic_safetensors(self.source / "model.safetensors", bad)
        with self.assertRaisesRegex(PruningError, "Unsupported MLP"):
            load_model_spec(self.source)
        bad = dict(self.arrays)
        bad["model.layers.1.mlp.up_proj.weight"] = np.zeros((5, 3))
        write_synthetic_safetensors(self.source / "model.safetensors", bad)
        with self.assertRaisesRegex(PruningError, "wrong FFN shape"):
            load_model_spec(self.source)

    def test_invalid_header_and_nonfinite_weights_rejected(self):
        path = self.source / "model.safetensors"
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaisesRegex(PruningError, "truncated"):
            SafeTensorFile(path)
        arrays = dict(self.arrays)
        arrays["model.layers.0.mlp.up_proj.weight"] = arrays["model.layers.0.mlp.up_proj.weight"].copy()
        arrays["model.layers.0.mlp.up_proj.weight"][0, 0] = np.nan
        write_synthetic_safetensors(path, arrays)
        output = self.root / "nonfinite-out"
        with self.assertRaisesRegex(PruningError, "Non-finite"):
            crop_model(self.source, output, target_intermediate_size=3, apply=True)
        self.assertFalse(output.exists())

    def test_invalid_target_rejected(self):
        for options in ({"keep_ratio": 1.0}, {"keep_ratio": 0}, {"keep_ratio": float("nan")},
                        {"target_intermediate_size": 0}, {"target_intermediate_size": 6},
                        {"target_intermediate_size": 7}, {"target_intermediate_size": True},
                        {"keep_ratio": 0.5, "target_intermediate_size": 3}):
            with self.subTest(options=options), self.assertRaises(PruningError):
                plan_pruning(self.source, **options)

    def calibration_fixture(self):
        return {"schema_version": 1, "score_method": "activation_weight_energy",
                "model_config_sha256": hashlib.sha256((self.source / "config.json").read_bytes()).hexdigest(),
                "model_weights_sha256": hashlib.sha256((self.source / "model.safetensors").read_bytes()).hexdigest(),
                "calibration_data_sha256": "a" * 64,
                "model_dimensions": {"num_hidden_layers": 2, "hidden_size": 3, "intermediate_size": 6},
                "layer_scores": {"0": [9.0, 0.0, 8.0, 0.0, 7.0, 0.0],
                                 "1": [0.0, 7.0, 0.0, 8.0, 0.0, 9.0]},
                "calibration": {"samples": 2, "tokens": 10, "max_tokens": 8}}

    def test_external_calibration_drives_selection_and_provenance(self):
        path = self.root / "scores.json"
        path.write_text(json.dumps(self.calibration_fixture()))
        output = self.root / "calibrated"
        manifest = crop_model(self.source, output, target_intermediate_size=3,
                              scores_json=path, apply=True)
        self.assertEqual([layer["kept_channel_indices"] for layer in manifest["layers"]],
                         [[0, 2, 4], [1, 3, 5]])
        self.assertEqual(manifest["score_method"], "activation_weight_energy")
        self.assertEqual(manifest["calibration_score_source"]["scores_file_sha256"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        cropped = SafeTensorFile(output / "model.safetensors")
        name = "model.layers.0.mlp.down_proj.weight"
        np.testing.assert_array_equal(as_float(cropped, name), self.arrays[name][:, [0, 2, 4]])

    def test_external_calibration_rejects_wrong_identity_dimensions_and_values(self):
        path = self.root / "bad-scores.json"
        invalids = []
        for field, value in (("schema_version", 2), ("score_method", "unknown"),
                             ("model_config_sha256", "b" * 64),
                             ("model_weights_sha256", "b" * 64),
                             ("model_weights_sha256", None),
                             ("model_dimensions", {"num_hidden_layers": 2, "hidden_size": 4, "intermediate_size": 6}),
                             ("layer_scores", {"0": [1.0] * 6}),
                             ("layer_scores", {"0": [1.0] * 5, "1": [1.0] * 6}),
                             ("layer_scores", {"0": [float("inf")] * 6, "1": [1.0] * 6}),
                             ("layer_scores", {"0": [-1.0] * 6, "1": [1.0] * 6}),
                             ("layer_scores", {"0": [True] * 6, "1": [1.0] * 6})):
            invalid = self.calibration_fixture()
            invalid[field] = value
            invalids.append(invalid)
        for invalid in invalids:
            with self.subTest(invalid=invalid):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(PruningError):
                    plan_pruning(self.source, target_intermediate_size=3, scores_json=path)

    def test_calibration_rejects_different_weights_with_identical_config(self):
        path = self.root / "scores-original-weights.json"
        path.write_text(json.dumps(self.calibration_fixture()))
        modified = dict(self.arrays)
        name = "model.layers.0.mlp.gate_proj.weight"
        modified[name] = modified[name] + np.float32(0.125)
        write_synthetic_safetensors(self.source / "model.safetensors", modified)
        with self.assertRaisesRegex(PruningError, "model_weights_sha256 does not match"):
            plan_pruning(self.source, target_intermediate_size=3, scores_json=path)

    def test_chunk_boundaries_produce_identical_checkpoint_bytes(self):
        hidden, intermediate = 129, 1031
        config = dict(self.config, hidden_size=hidden, intermediate_size=intermediate, head_dim=hidden)
        (self.source / "config.json").write_text(json.dumps(config))
        rng = np.random.default_rng(19)
        arrays = {"model.embed_tokens.weight": rng.normal(size=(7, hidden)).astype(np.float32),
                  "model.norm.weight": np.ones(hidden, dtype=np.float32),
                  "lm_head.weight": rng.normal(size=(7, hidden)).astype(np.float32)}
        for layer in range(2):
            prefix = f"model.layers.{layer}."
            arrays[prefix + "self_attn.q_proj.weight"] = rng.normal(size=(hidden, hidden)).astype(np.float32)
            arrays[prefix + "input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
            arrays[prefix + "mlp.gate_proj.weight"] = rng.normal(size=(intermediate, hidden)).astype(np.float32)
            arrays[prefix + "mlp.up_proj.weight"] = rng.normal(size=(intermediate, hidden)).astype(np.float32)
            arrays[prefix + "mlp.down_proj.weight"] = rng.normal(size=(hidden, intermediate)).astype(np.float32)
        write_synthetic_safetensors(self.source / "model.safetensors", arrays)
        for budget in (1, 4):
            crop_model(self.source, self.root / f"chunk-{budget}",
                       target_intermediate_size=517, chunk_mib=budget, apply=True)
        first = (self.root / "chunk-1" / "model.safetensors").read_bytes()
        second = (self.root / "chunk-4" / "model.safetensors").read_bytes()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Contracts for unchanged synthetic sources routed through existing formatters."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import format_bridge_controls as bridge


class FormatBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get("FORMAT_TEST_COMPONENT_ROOT", Path(__file__).resolve().parent))
        cls.paths = {"legacy_preparer": root / "prepare_evidence_cases.py",
                     "packer_file": root / "experiments/evidence_packing.py",
                     "reader_file": root / "experiments/evidence_reader.py"}
        cls.legacy, cls.packer = bridge.load_components(**cls.paths)
        cls.result = bridge.build_bridge(cls.legacy, cls.packer)

    def test_calls_the_exact_existing_formatters(self):
        with patch.object(self.legacy, "legacy_evidence", wraps=self.legacy.legacy_evidence) as old, \
                patch.object(self.packer, "pack_evidence", wraps=self.packer.pack_evidence) as pack:
            result = bridge.build_bridge(self.legacy, self.packer)
            self.assertEqual(old.call_count, 8)
            self.assertEqual(pack.call_count, 16)
            self.assertEqual(len(result["cases"]), 24)
            self.assertTrue(all(call.kwargs["max_chars"] == 8192 for call in pack.call_args_list))

    def test_same_sources_questions_and_rubrics_as_frozen_matrix(self):
        matrix = bridge.matrix.build_experiment()
        self.assertEqual(self.result["source_catalog"], matrix["source_catalog"])
        catalog = bridge.matrix.canonical_bytes({"synthetic": True, "scenarios": matrix["source_catalog"]}) + b"\n"
        self.assertEqual(self.result["source_catalog_bytes"], catalog)
        matrix_keys = {row["scenario_id"]: row for row in matrix["answer_key"]}
        for key in self.result["answer_key"]:
            other = matrix_keys[key["scenario_id"]]
            self.assertEqual({k: v for k, v in key.items() if k != "id"},
                             {k: v for k, v in other.items() if k != "id"})
        for case in self.result["cases"]:
            self.assertEqual(set(case), {"id", "question", "correct"})

    def test_complete_packs_and_single_name_intervention(self):
        cases = {row["id"]: row for row in self.result["cases"]}
        maps = {row["id"]: row for row in self.result["condition_map"]}
        sources = {row["scenario_id"]: row for row in self.result["source_catalog"]}
        for pair in self.result["contrasts"]:
            if pair["factor"] != "display_name_collision":
                self.assertEqual(pair["causal_interpretation"], "composite_format_bridge")
                continue
            left, right = cases[pair["left_id"]], cases[pair["right_id"]]
            self.assertEqual(left["question"], right["question"])
            source = sources[maps[left["id"]]["scenario_id"]]
            self.assertEqual(left["correct"][0], right["correct"][0])
            for message, a, b in zip(source["messages"], left["correct"][1:], right["correct"][1:]):
                # All body/quote lines are byte-identical; only the final name in
                # the existing one-line author header may change.
                a_lines, b_lines = a.splitlines(), b.splitlines()
                self.assertEqual(a_lines[1:], b_lines[1:])
                expected_a = bridge.matrix._name(message, source["collision_subject"], "neutral")
                expected_b = bridge.matrix._name(message, source["collision_subject"], "assistant")
                self.assertTrue(a_lines[0].endswith(json.dumps(expected_a, ensure_ascii=False)))
                self.assertTrue(b_lines[0].endswith(json.dumps(expected_b, ensure_ascii=False)))
                self.assertEqual(a_lines[0][:-len(json.dumps(expected_a, ensure_ascii=False))],
                                 b_lines[0][:-len(json.dumps(expected_b, ensure_ascii=False))])
        self.assertEqual(len(self.result["pack_audit"]), 16)
        for audit in self.result["pack_audit"]:
            for record in audit["pack"]["records"]:
                self.assertTrue(record["included"])
                self.assertTrue(all(field["complete"] and not field["omitted_ranges"] for field in record["fields"].values()))
        bad = deepcopy(self.result["pack_audit"][0]["pack"])
        bad["records"][0]["fields"]["body"]["complete"] = False
        mapping = maps[self.result["pack_audit"][0]["id"]]
        scenario = sources[mapping["scenario_id"]]
        index = [s["scenario_id"] for s in self.result["source_catalog"]].index(scenario["scenario_id"])
        messages = bridge.source_messages(scenario, index, mapping["factors"]["display_name_collision"])
        with self.assertRaises(ValueError):
            bridge.validate_complete_pack(bad, messages, self.result["synthetic_source_sha256"])

    def test_legacy_limit_and_normalization_checks(self):
        scenario = bridge.matrix.synthetic_scenarios()[0]
        original = bridge.source_messages(scenario, 0, "neutral")
        bridge.validate_legacy_sources(original)
        for field, value in (("body", "字" * 121), ("quoted", "字" * 61), ("body", "多  个空格")):
            bad = deepcopy(original); bad[0][field] = value
            with self.assertRaises(ValueError):
                bridge.validate_legacy_sources(bad)

    def test_component_identity_rejects_mismatched_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "changed_packer.py"
            path.write_text("raise RuntimeError('must never execute')\n")
            paths = dict(self.paths); paths["packer_file"] = path
            with self.assertRaisesRegex(ValueError, "component source differs"):
                bridge.load_components(**paths)

    def test_deterministic_files_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            first = bridge.write_bridge(a, **self.paths)
            second = bridge.write_bridge(b, **self.paths)
            self.assertEqual(first, second)
            self.assertEqual({p.name: p.read_bytes() for p in a.iterdir()},
                             {p.name: p.read_bytes() for p in b.iterdir()})
            self.assertEqual(first["cases"], 24)
            self.assertEqual(first["status"], "prepared_not_executed")
            self.assertFalse(first["model_generation_run"])
            self.assertIsNone(first["model_accuracy"])
            for name, meta in first["files"].items():
                raw = (a / name).read_bytes()
                self.assertEqual(len(raw), meta["bytes"])
                self.assertEqual(bridge.matrix.sha256(raw), meta["sha256"])
            with self.assertRaises(FileExistsError):
                bridge.write_bridge(a, **self.paths)


if __name__ == "__main__":
    unittest.main()

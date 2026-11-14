"""CPU-only contracts for paired synthetic inputs; no model-quality assertions."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments import format_controls as controls


def strip_lines(text, prefix):
    return "\n".join(line for line in text.splitlines() if not line.startswith(prefix))


def normalize_serialization(text, *, json_values):
    labels = ("显示名：", "正文：", "引用（不自动归属当前发送者）：")
    lines = []
    for line in text.splitlines():
        for label in labels:
            if line.startswith(label):
                value = line[len(label):]
                line = label + (json.loads(value) if json_values else value)
                break
        lines.append(line)
    return "\n".join(lines)


def move_author_before_body(text):
    lines = text.splitlines()
    start, end = lines.index("[作者]"), lines.index("[/作者]")
    block = lines[start:end + 1]
    del lines[start:end + 1]
    return "\n".join([lines[0], *block, *lines[1:]])


class FormatControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = controls.build_experiment()
        cls.cases = {r["id"]: r for r in cls.result["cases"]}
        cls.maps = {r["id"]: r for r in cls.result["condition_map"]}
        cls.sources = {r["scenario_id"]: r for r in cls.result["source_catalog"]}

    def pairs(self, factor):
        for pair in self.result["contrasts"]:
            if pair["factor"] == factor:
                yield self.cases[pair["left_id"]], self.cases[pair["right_id"]]

    def test_balanced_design_and_case_schema(self):
        self.assertEqual(len(self.result["cases"]), 136)
        self.assertEqual(Counter(m["scenario_id"] for m in self.maps.values()),
                         Counter({f"S{i:02d}": 17 for i in range(1, 9)}))
        self.assertEqual(Counter(p["factor"] for p in self.result["contrasts"]),
                         Counter({**{name: 64 for name in controls.FACTORS}, "display_name_presence": 8}))
        self.assertEqual(Counter(s["collision_subject"] for s in self.sources.values()),
                         Counter(self=4, other=4))
        self.assertTrue(all(set(row) == {"id", "question", "correct"} for row in self.cases.values()))
        self.assertTrue(all(len(s["messages"]) <= 3 for s in self.sources.values()))

    def test_answer_key_mutation_cannot_reach_model_inputs(self):
        scenarios = controls.synthetic_scenarios()
        for row in scenarios:
            row["expected_answer"] = "GOLD_ONLY_SENTINEL_DO_NOT_RENDER"
            row["requirements"] = ["RUBRIC_ONLY_SENTINEL_DO_NOT_RENDER"]
            row["forbidden_claims"] = ["GRADE_ONLY_SENTINEL_DO_NOT_RENDER"]
            row["answerable"] = not row["answerable"]
        altered = controls.build_experiment(scenarios)
        self.assertEqual(altered["cases"], self.result["cases"])
        self.assertEqual(altered["condition_map"], self.result["condition_map"])
        self.assertNotEqual(altered["answer_key"], self.result["answer_key"])
        payload = json.dumps(altered["cases"], ensure_ascii=False)
        self.assertNotIn("SENTINEL", payload)
        for factor in controls.FACTORS:
            self.assertNotIn(factor, payload)

    def test_serialization_pairs_recover_identical_original_values(self):
        escaped_quote_seen = False
        for left, right in self.pairs("string_serialization"):
            self.assertEqual(left["question"], right["question"])
            for plain, quoted in zip(left["correct"], right["correct"]):
                self.assertEqual(normalize_serialization(plain, json_values=False),
                                 normalize_serialization(quoted, json_values=True))
                escaped_quote_seen |= '\\"practice-' in quoted
        self.assertTrue(escaped_quote_seen)

    def test_collision_only_changes_name_for_the_selected_source_role(self):
        for left, right in self.pairs("display_name_collision"):
            info = self.maps[left["id"]]
            source = self.sources[info["scenario_id"]]
            for index, (normal, collision) in enumerate(zip(left["correct"], right["correct"])):
                self.assertEqual(strip_lines(normal, "显示名："), strip_lines(collision, "显示名："))
                subject = "self" if source["messages"][index]["is_self"] else "other"
                if subject == source["collision_subject"]:
                    self.assertNotEqual(normal, collision)
                    self.assertIn("assistant", collision)
                else:
                    self.assertEqual(normal, collision)

    def test_position_only_moves_the_unchanged_author_block(self):
        for left, right in self.pairs("author_block_position"):
            for prefix, suffix in zip(left["correct"], right["correct"]):
                self.assertEqual(prefix, move_author_before_body(suffix))

    def test_extra_boolean_field_preserves_authoritative_role(self):
        for left, right in self.pairs("extra_is_self_field"):
            source = self.sources[self.maps[left["id"]]["scenario_id"]]
            for message, plain, explicit in zip(source["messages"], left["correct"], right["correct"]):
                self.assertEqual(plain, strip_lines(explicit, "is_self："))
                expected = "is_self：true" if message["is_self"] else "is_self：false"
                self.assertIn(expected, explicit.splitlines())
                self.assertIn("归属：" + ("本人" if message["is_self"] else "他人"), plain.splitlines())

    def test_no_display_name_controls_remove_only_that_field(self):
        for left, right in self.pairs("display_name_presence"):
            for without, with_name in zip(left["correct"], right["correct"]):
                self.assertEqual(without, strip_lines(with_name, "显示名："))
                self.assertIn("[作者]", without)
                self.assertIn("归属：", without)

    def test_integrity_checks_reject_content_and_pair_tampering(self):
        for change in ("body", "factor", "duplicate"):
            bad = deepcopy(self.result)
            if change == "body":
                bad["cases"][0]["correct"][0] += "被追加的未声明事实。"
            elif change == "factor":
                bad["contrasts"][0]["factor"] = "unregistered_factor"
            else:
                bad["cases"].append(deepcopy(bad["cases"][0]))
            with self.assertRaises(ValueError):
                controls.validate_design(bad)

    def test_contrast_metadata_direction_and_exhaustiveness(self):
        for change in ("scenario", "left_level", "right_level", "fixed_factors", "direction", "duplicate"):
            bad = deepcopy(self.result)
            pair = bad["contrasts"][0]
            if change == "scenario":
                pair["scenario_id"] = "S99"
            elif change == "left_level":
                pair["left_level"] = "undeclared"
            elif change == "right_level":
                pair["right_level"] = "undeclared"
            elif change == "fixed_factors":
                pair["fixed_factors"] = {}
            elif change == "direction":
                pair["left_id"], pair["right_id"] = pair["right_id"], pair["left_id"]
                pair["left_level"], pair["right_level"] = pair["right_level"], pair["left_level"]
            else:
                bad["contrasts"][1] = deepcopy(pair)
            with self.assertRaises(ValueError, msg=change):
                controls.validate_design(bad)

    def test_deterministic_artifacts_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            reader = Path(temp) / "reader_fixture.py"
            reader.write_text("# Synthetic file-identity fixture; never executed.\n")
            first = controls.write_experiment(a, reader_file=reader)
            second = controls.write_experiment(b, reader_file=reader)
            self.assertEqual(first, second)
            self.assertEqual({p.name: p.read_bytes() for p in a.iterdir()},
                             {p.name: p.read_bytes() for p in b.iterdir()})
            self.assertEqual(first["status"], "prepared_not_executed")
            self.assertFalse(first["model_generation_run"])
            self.assertIsNone(first["model_accuracy"])
            self.assertIsNone(first["actual_input_tokens"])
            self.assertEqual(first["reader_sha256"], controls.sha256(reader.read_bytes()))
            for name, meta in first["files"].items():
                raw = (a / name).read_bytes()
                self.assertEqual(len(raw), meta["bytes"])
                self.assertEqual(controls.sha256(raw), meta["sha256"])
            with self.assertRaises(FileExistsError):
                controls.write_experiment(a, reader_file=reader)


if __name__ == "__main__":
    unittest.main()

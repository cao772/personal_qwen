"""CPU-only contract tests with fabricated output rows; never model results."""
from copy import deepcopy
import json
from pathlib import Path
import stat
import tempfile
import unittest

import audit_format_run as audit
from experiments import evidence_reader
from experiments.format_controls import write_experiment


class FormatRunAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.fixture.name)
        cls.reader = Path(evidence_reader.__file__)
        cls.bundle = cls.fixture_root / "synthetic_bundle"
        write_experiment(cls.bundle, reader_file=cls.reader)
        cls.manifest_hash = audit.sha256((cls.bundle / "manifest.json").read_bytes())
        cls.cases = [json.loads(line) for line in (cls.bundle / "cases.jsonl").read_text().splitlines()]
        case_hash = audit.sha256((cls.bundle / "cases.jsonl").read_bytes())
        cls.mock_rows = []
        for case in cls.cases:
            messages = evidence_reader.chat_messages(case, case["correct"], "research", "original")
            cls.mock_rows.append({
                "id": case["id"], "condition": "correct",
                "answer": "完全合成的模拟输出，不是模型回答。", "tokens": 8,
                "elapsed": 0.0, "hit_cap": False,
                **evidence_reader.request_audit(case["correct"], messages, [101, 102, 103],
                    mode="research", prompt_version="original", max_input_tokens=4096),
                "max_new_tokens": 192, "case_file_sha256": case_hash,
            })

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.serial = 0

    def run_audit(self, rows=None, *, raw=None, **overrides):
        self.serial += 1
        answers_file = self.root / f"mock_answers_{self.serial}.jsonl"
        if raw is None:
            raw = b"".join(audit.canonical_bytes(row) + b"\n"
                           for row in (self.mock_rows if rows is None else rows))
        answers_file.write_bytes(raw)
        options = dict(bundle=self.bundle, manifest_sha256=self.manifest_hash,
                       reader_file=self.reader, answers_file=answers_file,
                       out=self.root / f"audit_{self.serial}")
        options.update(overrides)
        result = audit.audit_run(**options)
        return result, Path(options["out"])

    def test_valid_binding_blinding_determinism_and_private_permissions(self):
        rows = deepcopy(self.mock_rows)
        rows[0].update(tokens=192, hit_cap=True)
        report, out = self.run_audit(rows)
        _, repeated = self.run_audit(list(reversed(rows)))
        self.assertEqual(report["records"], 136)
        self.assertEqual(report["output_cap_records"], 1)
        self.assertFalse(report["input_token_ids_sha256_recomputed"])
        self.assertFalse(report["model_execution_verified"])
        self.assertFalse(report["semantic_scoring_performed"])
        self.assertEqual((out / "blind_cases.jsonl").read_bytes(),
                         (repeated / "blind_cases.jsonl").read_bytes())
        key = json.loads((out / "blind_key.json").read_bytes())
        by_blind = {row["blind_id"]: row for row in key["mapping"]}
        self.assertEqual(len(by_blind), report["records"])
        cases = {row["id"]: row for row in self.cases}
        rubrics = {row["id"]: row for row in
                   [json.loads(line) for line in (self.bundle / "answer_key.jsonl").read_text().splitlines()]}
        for blind in [json.loads(line) for line in (out / "blind_cases.jsonl").read_text().splitlines()]:
            self.assertEqual(set(blind), {"blind_id", "question", "evidence", "answer", "answerable", "rubric", "generation"})
            original = cases[by_blind[blind["blind_id"]]["id"]]
            self.assertEqual(blind["question"], original["question"])
            self.assertEqual(blind["evidence"], original["correct"])
            self.assertEqual(blind["answer"], "完全合成的模拟输出，不是模型回答。")
            rubric = rubrics[original["id"]]
            self.assertEqual(blind["answerable"], rubric["answerable"])
            self.assertEqual(blind["rubric"], {name: rubric[name] for name in
                             ("expected_answer", "requirements", "forbidden_claims", "grading_rule")})
        self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o700)
        for name in ("request_audit.json", "blind_cases.jsonl", "blind_key.json"):
            self.assertEqual(stat.S_IMODE((out / name).stat().st_mode), 0o600)

    def test_missing_duplicate_and_extra_ids_rejected(self):
        extra = dict(self.mock_rows[0], id="synthetic-extra-id")
        for rows in (self.mock_rows[:-1], self.mock_rows + [self.mock_rows[0]], self.mock_rows + [extra]):
            with self.subTest(count=len(rows)), self.assertRaises(audit.AuditError):
                self.run_audit(rows)
        self.assertFalse(any(self.root.glob("audit_*")))

    def test_request_hash_prompt_and_condition_mismatches_rejected(self):
        mutations = {
            "case_file_sha256": "0" * 64, "evidence_sha256": "0" * 64,
            "chat_messages_sha256": "0" * 64, "requested_prompt_version": "checked",
            "effective_prompt_version": "chat", "condition": "oracle",
            "hash_format": "unknown", "input_token_ids_sha256": "not-a-digest",
        }
        for name, value in mutations.items():
            rows = deepcopy(self.mock_rows)
            rows[0][name] = value
            with self.subTest(field=name), self.assertRaises(audit.AuditError):
                self.run_audit(rows)

    def test_changed_reader_or_external_manifest_binding_rejected(self):
        changed_reader = self.root / "reader.py"
        changed_reader.write_bytes(self.reader.read_bytes() + b"\n# synthetic change\n")
        with self.assertRaises(audit.AuditError):
            self.run_audit(reader_file=changed_reader)
        with self.assertRaises(audit.AuditError):
            self.run_audit(manifest_sha256="0" * 64)
        changed_bundle = self.root / "changed_manifest_bundle"
        changed_bundle.mkdir()
        for source in self.bundle.iterdir():
            (changed_bundle / source.name).write_bytes(source.read_bytes())
        path = changed_bundle / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["reader_sha256"] = audit.sha256(changed_reader.read_bytes())
        path.write_bytes(audit.canonical_bytes(manifest) + b"\n")
        with self.assertRaises(audit.AuditError):
            self.run_audit(bundle=changed_bundle)
        with self.assertRaises(audit.AuditError):
            self.run_audit(bundle=changed_bundle, reader_file=changed_reader,
                           manifest_sha256=audit.sha256(path.read_bytes()))

    def test_caps_counts_and_flags_rejected_when_inconsistent(self):
        mutations = ({"input_tokens": 4097}, {"input_tokens": True},
                     {"max_input_tokens": None}, {"max_new_tokens": 193},
                     {"tokens": 193, "hit_cap": True}, {"tokens": 192, "hit_cap": False},
                     {"tokens": 8, "hit_cap": True}, {"tokens": 0},
                     {"elapsed": -1}, {"hit_cap": 0})
        for mutation in mutations:
            rows = deepcopy(self.mock_rows)
            rows[0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(audit.AuditError):
                self.run_audit(rows)

    def test_unknown_missing_error_fields_and_malformed_records_rejected(self):
        rows = deepcopy(self.mock_rows)
        rows[0]["error"] = "synthetic simulated generation error"
        with self.assertRaises(audit.AuditError):
            self.run_audit(rows)
        rows = deepcopy(self.mock_rows)
        del rows[0]["input_tokens"]
        with self.assertRaises(audit.AuditError):
            self.run_audit(rows)
        valid = b"".join(audit.canonical_bytes(row) + b"\n" for row in self.mock_rows)
        for raw in (valid + b"\n", valid.replace(b'"tokens":8', b'"tokens":8,"tokens":8', 1),
                    valid.replace(b'"elapsed":0.0', b'"elapsed":NaN', 1), valid + b'{"broken":'):
            with self.subTest(size=len(raw)), self.assertRaises(audit.AuditError):
                self.run_audit(raw=raw)

    def test_bundle_evidence_bytes_tampering_rejected(self):
        changed = self.root / "changed_bundle"
        changed.mkdir()
        for source in self.bundle.iterdir():
            (changed / source.name).write_bytes(source.read_bytes())
        path = changed / "cases.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(audit.AuditError):
            self.run_audit(bundle=changed)

    def test_existing_output_directory_is_not_overwritten(self):
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "sentinel.txt"
        sentinel.write_text("synthetic existing file")
        with self.assertRaises(audit.AuditError):
            self.run_audit(out=existing)
        self.assertEqual(list(existing.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), "synthetic existing file")


if __name__ == "__main__":
    unittest.main()

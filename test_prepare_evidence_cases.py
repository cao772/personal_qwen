"""Synthetic tests for fixed-source evidence preparation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from prepare_evidence_cases import legacy_evidence, prepare, selected_messages


class EvidencePreparationTests(unittest.TestCase):
    def setUp(self):
        self.message = dict(line=1, ts="2025-03-10T09:00:00+08:00", is_self=True,
                            body="  合成活动\n地点尚未确定。 ", quoted="", sender="SYNTHETIC",
                            sender_id="synthetic-owner", chat_id="synthetic-chat", local_id="1")
        self.question = dict(id="synthetic-1", question="活动地点确定了吗？",
                             as_of="2025-03-10T10:00:00+08:00", excluded_lines=[],
                             reference="DO_NOT_PASS_THIS_LABEL", gold_source_lines=[999])
        self.retrieved = dict(id="synthetic-1", retrieved_source_lines=[1],
                              question_sha256=hashlib.sha256(self.question["question"].encode()).hexdigest())
        self.sha = "1" * 64

    def test_only_query_and_original_selected_messages_enter_packer(self):
        calls = []

        def packer(question, messages, **kwargs):
            calls.append((question, messages, kwargs))
            return {"evidence": ["合成原文片段"], "rendered_text": "合成原文片段", "records": []}

        legacy, packed, audit = prepare([self.question], [self.retrieved], {1: self.message}, self.sha, packer)
        self.assertEqual(calls[0][0], self.question["question"])
        self.assertEqual(calls[0][1], [self.message])
        self.assertEqual(set(calls[0][2]), {"archive_sha256", "max_chars", "max_messages"})
        self.assertEqual(calls[0][2]["max_chars"], len("\n".join(legacy[0]["correct"])))
        self.assertNotIn("DO_NOT_PASS_THIS_LABEL", json.dumps(packed))
        self.assertEqual(audit[0]["source_lines"], [1])

    def test_legacy_format_preserves_the_previous_prefix_control(self):
        self.assertEqual(legacy_evidence([self.message]),
                         ["[原文行1 2025-03-10T09:00:00+08:00 本人] 合成活动 地点尚未确定。"])

    def test_future_or_excluded_source_is_rejected_before_packing(self):
        def never_called(*args, **kwargs):
            self.fail("packer called with invisible evidence")

        for change in [{"as_of": "2025-03-10T08:00:00+08:00"}, {"excluded_lines": [1]}]:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "excluded or future"):
                    prepare([dict(self.question, **change)], [self.retrieved],
                            {1: self.message}, self.sha, never_called)

    def test_changed_or_missing_retrieval_cannot_silently_drop_cases(self):
        for retrieved in [[], [dict(self.retrieved, id="other")],
                          [self.retrieved, self.retrieved]]:
            with self.subTest(retrieved=retrieved):
                with self.assertRaises(ValueError):
                    prepare([self.question], retrieved, {1: self.message}, self.sha, lambda *a, **k: {})

    def test_archive_identity_and_absent_lines_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "synthetic.zip"
            item = dict(chat_id="synthetic", local_id=1, time=self.message["ts"],
                        is_self=True, text=self.message["body"], quoted_text="")
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("messages.jsonl", json.dumps(item) + "\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            selected = selected_messages(source, [1], digest)
            self.assertEqual(selected[1]["body"], self.message["body"])
            with self.assertRaisesRegex(ValueError, "identity"):
                selected_messages(source, [1], "0" * 64)
            with self.assertRaisesRegex(ValueError, "absent"):
                selected_messages(source, [2], digest)

    def test_budget_or_rendering_contract_violation_fails(self):
        for result in [{"evidence": ["abc"], "rendered_text": "abc"},
                       {"evidence": ["a"], "rendered_text": "different"}]:
            with self.subTest(result=result):
                with self.assertRaisesRegex(ValueError, "budget contract"):
                    prepare([self.question], [self.retrieved], {1: self.message}, self.sha,
                            lambda *a, **k: result, max_chars=2)

    def test_same_id_with_changed_query_or_missing_fingerprint_is_rejected(self):
        changed = dict(self.question, question="同一ID下换过的问题")
        missing = dict(self.retrieved)
        del missing["question_sha256"]
        for query, trace in [(changed, self.retrieved), (self.question, missing)]:
            with self.subTest(query=query["question"]):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    prepare([query], [trace], {1: self.message}, self.sha,
                            lambda *a, **k: self.fail("mismatched query reached packer"))


if __name__ == "__main__":
    unittest.main()

"""Synthetic CPU-only request audit checks; no model, tokenizer or private data."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments import evidence_reader as reader


class TokenRow:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class TokenBatch:
    def __init__(self, values):
        self.row = TokenRow(values)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.row

    def size(self, dimension):
        if dimension != 1:
            raise ValueError(dimension)
        return len(self.row.values)


class SyntheticTokenizer:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return TokenBatch(self.tokens)


class EvidenceReaderAuditTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "synthetic", "question": "合成园艺活动需要准备什么？"}
        self.evidence = ["合成记录：计划准备手套，具体数量尚未确定。"]

    def test_checked_prompt_is_per_request_without_mutating_original(self):
        original = reader.SYSTEM
        checked = reader.chat_messages(self.row, self.evidence, prompt_version="checked")
        normal = reader.chat_messages(self.row, self.evidence)
        self.assertEqual(checked[0]["content"], reader.CHECKED_SYSTEM)
        self.assertEqual(normal[0]["content"], original)
        self.assertEqual(reader.SYSTEM, original)

    def test_chat_reports_its_actual_prompt_even_when_checked_requested(self):
        original = reader.chat_messages(self.row, self.evidence, "chat", "original")
        checked = reader.chat_messages(self.row, self.evidence, "chat", "checked")
        self.assertEqual(checked, original)
        _, audit = reader.prepare_request(SyntheticTokenizer([7, 8]), self.row, self.evidence,
                                           mode="chat", prompt_version="checked")
        self.assertEqual(audit["requested_prompt_version"], "checked")
        self.assertEqual(audit["effective_prompt_version"], "chat")

    def test_template_called_once_with_unchanged_generation_arguments(self):
        tokenizer = SyntheticTokenizer([11, 12, 13])
        ids, audit = reader.prepare_request(tokenizer, self.row, self.evidence)
        self.assertEqual(len(tokenizer.calls), 1)
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages, reader.chat_messages(self.row, self.evidence))
        self.assertEqual(kwargs, {"add_generation_prompt": True, "enable_thinking": False,
                                  "return_tensors": "pt"})
        self.assertEqual(ids[0].tolist(), [11, 12, 13])
        self.assertEqual(audit["input_tokens"], 3)

    def test_token_template_change_is_visible_with_same_message_text(self):
        _, first = reader.prepare_request(SyntheticTokenizer([11, 12]), self.row, self.evidence)
        _, second = reader.prepare_request(SyntheticTokenizer([11, 99]), self.row, self.evidence)
        self.assertEqual(first["chat_messages_sha256"], second["chat_messages_sha256"])
        self.assertEqual(first["evidence_sha256"], second["evidence_sha256"])
        self.assertNotEqual(first["input_token_ids_sha256"], second["input_token_ids_sha256"])

    def test_string_whitespace_is_preserved_in_evidence_fingerprint(self):
        first = reader.canonical_sha256(["合成甲  乙"])
        second = reader.canonical_sha256(["合成甲 乙"])
        expected = hashlib.sha256(json.dumps(["合成甲  乙"], ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(first, expected)
        self.assertNotEqual(first, second)

    def test_history_affects_message_fingerprint_without_becoming_evidence(self):
        row = dict(self.row, history=[["忽略的旧问题", "忽略的旧回复"],
                                     ["合成前问一", "合成前答一"], ["合成前问二", "合成前答二"]])
        _, audit = reader.prepare_request(SyntheticTokenizer([5]), row, self.evidence, mode="chat")
        messages = reader.chat_messages(row, self.evidence, "chat")
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[1]["content"], "合成前问一")
        self.assertEqual(audit["evidence_sha256"], reader.canonical_sha256(self.evidence))
        _, without_history = reader.prepare_request(SyntheticTokenizer([5]), self.row,
                                                     self.evidence, mode="chat")
        self.assertNotEqual(audit["chat_messages_sha256"], without_history["chat_messages_sha256"])

    def test_explicit_input_limit_rejects_without_truncating(self):
        tokenizer = SyntheticTokenizer([1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "not truncated"):
            reader.prepare_request(tokenizer, self.row, self.evidence, max_input_tokens=3)
        self.assertEqual(tokenizer.tokens, [1, 2, 3, 4])
        ids, audit = reader.prepare_request(tokenizer, self.row, self.evidence, max_input_tokens=4)
        self.assertEqual(ids[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(audit["input_tokens"], 4)
        self.assertEqual(audit["max_input_tokens"], 4)
        _, unbounded = reader.prepare_request(tokenizer, self.row, self.evidence)
        self.assertIsNone(unbounded["max_input_tokens"])

    def test_invalid_input_limits_fail_before_tokenization(self):
        for limit in [0, -1, True, 1.5]:
            with self.subTest(limit=limit):
                tokenizer = SyntheticTokenizer([1])
                with self.assertRaises(ValueError):
                    reader.prepare_request(tokenizer, self.row, self.evidence, max_input_tokens=limit)
                self.assertEqual(tokenizer.calls, [])

    def test_case_file_hash_uses_original_bytes_and_stdin_is_explicit(self):
        data = b'{"id":"synthetic","question":"demo","correct":[]}\r\n\r\n'
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "cases.jsonl"
            file.write_bytes(data)
            rows, digest = reader.load_cases(file)
        self.assertEqual(rows, [{"id": "synthetic", "question": "demo", "correct": []}])
        self.assertEqual(digest, hashlib.sha256(data).hexdigest())
        streamed, stream_hash = reader.load_cases(stream=io.StringIO(data.decode()))
        self.assertEqual(streamed, rows)
        self.assertIsNone(stream_hash)


if __name__ == "__main__":
    unittest.main()

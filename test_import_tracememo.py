"""Synthetic TraceMemo rows only; no real identities, histories, or statistics."""

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from import_tracememo import ImportValidationError, import_tracememo


def fixture():
    return [
        {"chat_id": "synthetic-chat-a", "chat_name": "SYNTHETIC CHAT", "chat_type": "group",
         "sender": "SYNTHETIC SELF", "sender_id": "synthetic-self", "is_self": True,
         "time": "2030-01-02T10:00:00+08:00", "local_id": 1, "server_id": "synthetic-server-1",
         "type": "普通文本", "text": "SYNTHETIC QUESTION?", "future_metadata": {"keep": [1, None]}},
        {"chat_id": "synthetic-chat-a", "chat_name": "SYNTHETIC CHAT", "chat_type": "group",
         "sender": "SYNTHETIC OTHER", "sender_id": "synthetic-other", "is_self": False,
         "time": "2030-01-01T21:30:00-05:00", "local_id": 2, "server_id": "synthetic-server-2",
         "type": "引用消息", "text": "SYNTHETIC REPLY", "quoted_text": "SYNTHETIC QUOTE IS NOT AN OWNER FACT"},
    ]


class TraceMemoImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, rows=None, name="messages.jsonl"):
        path = self.root / name
        rows = fixture() if rows is None else rows
        path.write_bytes(b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows))
        return path

    def records(self, output):
        return [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]

    def assert_failed(self, output):
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(manifest["not_full_account_export_verification"])

    def test_source_identity_quote_and_unknown_fields_are_preserved(self):
        rows = fixture()
        source = self.source(rows)
        original = source.read_bytes()
        output = self.root / "imported"
        manifest = import_tracememo(source, output)
        records = self.records(output)
        self.assertEqual([row["speaker_role"] for row in records], ["self", "other"])
        self.assertEqual([row["is_self"] for row in records], [True, False])
        self.assertTrue(all(row["speaker_role"] not in ("user", "assistant") for row in records))
        self.assertEqual(records[0]["raw_record"], rows[0])
        self.assertEqual(records[1]["raw_record"], rows[1])
        self.assertEqual(records[1]["quoted_text"], rows[1]["quoted_text"])
        self.assertEqual(records[1]["text_index"], rows[1]["text"])
        self.assertNotIn(rows[1]["quoted_text"], records[1]["text_index"])
        self.assertEqual(records[1]["quote_interpretation"], "unparsed_quotation_not_owner_fact")
        self.assertEqual(records[1]["source_ref"]["line"], 2)
        self.assertEqual(records[0]["source_ref"]["line_sha256"], hashlib.sha256(original.splitlines(keepends=True)[0]).hexdigest())
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual((output / "source_original.jsonl").read_bytes(), original)
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(manifest["source"]["jsonl_sha256"], hashlib.sha256(original).hexdigest())
        self.assertFalse(manifest["interpretation"]["is_sft_dataset"])
        self.assertFalse(manifest["interpretation"]["facts_inferred"])

    def test_statistics_compare_explicit_timezones_in_utc(self):
        manifest = import_tracememo(self.source(), self.root / "imported")
        self.assertEqual(manifest["counts"]["rows"], 2)
        self.assertEqual(manifest["counts"]["chats"], 1)
        self.assertEqual(manifest["counts"]["self_rows"], 1)
        self.assertEqual(manifest["counts"]["other_rows"], 1)
        self.assertEqual(manifest["counts"]["time_min_utc"], "2030-01-02T02:00:00+00:00")
        self.assertEqual(manifest["counts"]["time_max_utc"], "2030-01-02T02:30:00+00:00")
        self.assertEqual(manifest["counts"]["message_types"], {"普通文本": 1, "引用消息": 1})

    def test_duplicate_chat_local_id_rejected_without_deduplication(self):
        rows = fixture()
        rows[1]["local_id"] = rows[0]["local_id"]
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "duplicate_chat_id_local_id"):
            import_tracememo(self.source(rows), output)
        self.assert_failed(output)

    def test_same_local_id_in_different_chats_is_valid(self):
        rows = fixture()
        rows[1]["chat_id"] = "synthetic-chat-b"
        rows[1]["local_id"] = 1
        manifest = import_tracememo(self.source(rows), self.root / "imported")
        self.assertEqual(manifest["counts"]["chats"], 2)

    def test_blank_text_and_sender_names_are_kept_without_identity_guessing(self):
        rows = fixture()
        rows[0]["text"] = ""
        rows[0]["sender"] = "  "
        rows[1]["text"] = " \n "
        rows[1]["sender"] = ""
        output = self.root / "imported"
        manifest = import_tracememo(self.source(rows), output)
        records = self.records(output)
        self.assertEqual(manifest["counts"]["rows"], 2)
        self.assertEqual(manifest["counts"]["blank_text_rows"], 2)
        self.assertEqual(manifest["counts"]["blank_sender_display_name_rows"], 2)
        self.assertEqual(records[0]["text_index"], "")
        self.assertEqual(records[1]["text_index"], " \n ")
        self.assertEqual(records[0]["author"]["display_name"], "  ")
        self.assertEqual(records[1]["author"]["id"], rows[1]["sender_id"])
        self.assertEqual([record["speaker_role"] for record in records], ["self", "other"])

    def test_timestamp_and_field_types_are_strict(self):
        invalid = [("is_self", 1), ("local_id", True), ("time", None),
                   ("time", "2030-01-02T10:00:00"), ("time", "not-a-time"),
                   ("quoted_text", {"not": "a string"}), ("sender_id", None)]
        for index, (field, value) in enumerate(invalid):
            with self.subTest(field=field, case=index):
                rows = fixture()
                rows[0][field] = value
                output = self.root / f"imported-{index}"
                with self.assertRaises(ImportValidationError):
                    import_tracememo(self.source(rows, f"input-{index}.jsonl"), output)
                self.assert_failed(output)

    def test_zip_preserved_and_member_paths_never_extracted(self):
        jsonl = self.source().read_bytes()
        source = self.root / "export.zip"
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("nested/messages.jsonl", jsonl)
            archive.writestr("../../escape.txt", "SYNTHETIC NEVER EXTRACT")
            archive.writestr("attachments/asset.bin", b"SYNTHETIC ASSET")
        original = source.read_bytes()
        output = self.root / "imported"
        manifest = import_tracememo(source, output)
        self.assertEqual((output / "source_original.zip").read_bytes(), original)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(manifest["source"]["jsonl_archive_member"], "nested/messages.jsonl")
        self.assertEqual({p.name for p in output.iterdir()}, {"source_original.zip", "manifest.json", "records.jsonl"})
        self.assertFalse((self.root / "escape.txt").exists())
        self.assertFalse((output / "attachments").exists())

    def test_zip_ambiguous_candidates_rejected(self):
        source = self.root / "export.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("messages.jsonl", b"{}\n")
            archive.writestr("backup/messages.jsonl", b"{}\n")
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "ambiguous"):
            import_tracememo(source, output)
        self.assert_failed(output)

    def test_explicit_limits_fail_without_silent_truncation(self):
        source = self.source()
        settings = [{"max_jsonl_bytes": 10}, {"max_line_bytes": 20}, {"max_records": 1}]
        for index, options in enumerate(settings):
            with self.subTest(options=options):
                output = self.root / f"imported-{index}"
                with self.assertRaisesRegex(ImportValidationError, "limit_exceeded_no_truncation"):
                    import_tracememo(source, output, **options)
                self.assert_failed(output)
                self.assertEqual((output / "source_original.jsonl").read_bytes(), source.read_bytes())

    def test_blank_line_and_duplicate_keys_are_not_silently_accepted(self):
        variants = [b"\n", b'{"chat_id":"one","chat_id":"two"}\n']
        for index, data in enumerate(variants):
            source = self.root / f"invalid-{index}.jsonl"
            source.write_bytes(data)
            output = self.root / f"imported-{index}"
            with self.assertRaises(ImportValidationError):
                import_tracememo(source, output)
            self.assert_failed(output)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits only")
    def test_private_permissions_and_original_input_mode(self):
        source = self.source()
        source_mode = stat.S_IMODE(source.stat().st_mode)
        output = self.root / "imported"
        old = os.umask(0o022)
        try:
            import_tracememo(source, output)
            self.assertEqual(os.umask(0o022), 0o022)
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir()))
        self.assertEqual(stat.S_IMODE(source.stat().st_mode), source_mode)

    def test_existing_output_is_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("SYNTHETIC SENTINEL")
        with self.assertRaises(FileExistsError):
            import_tracememo(self.source(), output)
        self.assertEqual(sentinel.read_text(), "SYNTHETIC SENTINEL")
        self.assertEqual(list(output.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()

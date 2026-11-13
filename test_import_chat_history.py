"""Synthetic-only importer tests; no account data or external services used."""

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from import_chat_history import ImportValidationError, import_history


def fixture():
    return [{
        "id": "conversation-1", "title": "SYNTHETIC PRIVATE TITLE",
        "create_time": 10, "update_time": None, "current_node": "assistant-a",
        "unknown_conversation_metadata": {"retained_in_original": True},
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["user"], "message": None},
            "user": {"id": "user", "parent": "root", "children": ["assistant-a", "assistant-b"],
                     "message": {"id": "message-user", "author": {"role": "user", "unknown": [1, 2]},
                                 "create_time": None,
                                 "content": {"content_type": "multimodal_text", "parts": [
                                     "SYNTHETIC QUESTION: does this mean I like blue?",
                                     {"content_type": "image_asset_pointer", "asset_pointer": "synthetic-image", "width": 20},
                                 ]},
                                 "metadata": {"unknown_future_field": {"nested": [False, None, "keep"]}}}},
            "assistant-a": {"id": "assistant-a", "parent": "user", "children": [],
                            "message": {"id": "message-a", "author": {"role": "assistant"},
                                        "create_time": 11,
                                        "content": {"content_type": "text", "parts": ["SYNTHETIC ASSISTANT CLAIM"]},
                                        "metadata": {"model": "synthetic"}}},
            "assistant-b": {"id": "assistant-b", "parent": "user", "children": ["tool"],
                            "message": {"id": "message-b", "author": {"role": "assistant"},
                                        "content": {"content_type": "text", "parts": ["SYNTHETIC OTHER BRANCH"]}}},
            "tool": {"id": "tool", "parent": "assistant-b", "children": [],
                     "message": {"id": "message-tool", "author": {"role": "tool", "name": "synthetic_tool"},
                                 "content": {"content_type": "code", "text": "SYNTHETIC TOOL RESULT"}}},
        },
    }]


class ImportHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, data=None, name="conversations.json"):
        path = self.root / name
        path.write_bytes(json.dumps(fixture() if data is None else data, ensure_ascii=False).encode("utf-8"))
        return path

    def records(self, output):
        return [json.loads(line) for line in (output / "messages.jsonl").read_text().splitlines()]

    def assert_failed(self, output):
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(manifest["not_full_account_export_verification"])

    def test_all_branches_roles_missing_times_and_raw_content(self):
        data = fixture()
        source = self.source(data)
        output = self.root / "imported"
        manifest = import_history(source, output)
        records = {row["node_id"]: row for row in self.records(output)}
        self.assertEqual(len(records), 5)
        self.assertEqual(manifest["counts"]["message_nodes"], 4)
        self.assertEqual(manifest["counts"]["nodes_without_message"], 1)
        self.assertEqual(manifest["counts"]["non_current_branch_message_nodes"], 2)
        self.assertTrue(records["root"]["current_path"])
        self.assertTrue(records["assistant-a"]["current_path"])
        self.assertFalse(records["assistant-b"]["current_path"])
        self.assertFalse(records["tool"]["current_path"])
        self.assertEqual(records["user"]["role"], "user")
        self.assertEqual(records["assistant-a"]["role"], "assistant")
        self.assertEqual(records["tool"]["author"]["name"], "synthetic_tool")
        self.assertIsNone(records["assistant-b"]["timestamps"]["message_create_time"])
        self.assertEqual(records["user"]["raw_node"], data[0]["mapping"]["user"])
        self.assertEqual(records["user"]["content"], data[0]["mapping"]["user"]["message"]["content"])
        self.assertEqual(records["user"]["metadata"], data[0]["mapping"]["user"]["message"]["metadata"])
        self.assertNotIn("synthetic-image", records["user"]["text_index"])
        self.assertEqual(records["assistant-a"]["interpretation_status"], "raw_utterance_no_fact_or_training_label_inferred")
        self.assertFalse(manifest["interpretation"]["training_labels_generated"])
        self.assertFalse(manifest["interpretation"]["user_facts_inferred"])

    def test_source_unchanged_and_complete_copy_hash(self):
        source = self.source()
        before = source.read_bytes()
        output = self.root / "imported"
        manifest = import_history(source, output)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual((output / "source_original.json").read_bytes(), before)
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(manifest["status"], "complete_for_provided_file")
        self.assertEqual(manifest["coverage_claim"], "not_full_account_export_verification")

    def test_zip_is_preserved_without_extracting_paths(self):
        source = self.root / "export.zip"
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("export/conversations.json", json.dumps(fixture()))
            archive.writestr("../../escaped.txt", "SYNTHETIC FILE MUST NOT BE EXTRACTED")
            archive.writestr("assets/image.bin", b"synthetic attachment bytes")
        before = source.read_bytes()
        output = self.root / "imported"
        manifest = import_history(source, output)
        self.assertEqual((output / "source_original.zip").read_bytes(), before)
        self.assertEqual(manifest["source"]["json_archive_member"], "export/conversations.json")
        self.assertEqual({path.name for path in output.iterdir()}, {"source_original.zip", "manifest.json", "messages.jsonl"})
        self.assertFalse((self.root / "escaped.txt").exists())
        self.assertFalse((output / "assets").exists())
        self.assertFalse((output / "export").exists())

    def test_ambiguous_zip_candidates_fail(self):
        source = self.root / "export.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("conversations.json", json.dumps(fixture()))
            archive.writestr("backup/conversations.json", json.dumps(fixture()))
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "ambiguous"):
            import_history(source, output)
        self.assert_failed(output)

    def test_duplicate_conversation_id_fails(self):
        data = fixture()
        data.append(copy.deepcopy(data[0]))
        source = self.source(data)
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "duplicate_conversation_id"):
            import_history(source, output)
        self.assert_failed(output)

    def test_duplicate_json_mapping_key_fails_without_silent_overwrite(self):
        source = self.root / "conversations.json"
        source.write_text('[{"id":"c","mapping":{"n":{},"n":{}}}]')
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "duplicate_json_object_key"):
            import_history(source, output)
        self.assert_failed(output)

    def test_duplicate_message_id_fails(self):
        data = fixture()
        data[0]["mapping"]["assistant-b"]["message"]["id"] = "message-a"
        source = self.source(data)
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "duplicate_message_id"):
            import_history(source, output)
        self.assert_failed(output)

    def test_invalid_ids_and_node_id_mismatch_fail(self):
        for index, change in enumerate(("blank_conversation", "numeric_message", "node_mismatch")):
            with self.subTest(change=change):
                data = fixture()
                if change == "blank_conversation":
                    data[0]["id"] = " "
                elif change == "numeric_message":
                    data[0]["mapping"]["user"]["message"]["id"] = 123
                else:
                    data[0]["mapping"]["user"]["id"] = "another-node"
                source = self.source(data, f"source-{index}.json")
                output = self.root / f"imported-{index}"
                with self.assertRaises(ImportValidationError):
                    import_history(source, output)
                self.assert_failed(output)

    def test_json_limit_fails_instead_of_truncating(self):
        source = self.source()
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "json_size_limit"):
            import_history(source, output, max_json_bytes=source.stat().st_size - 1)
        self.assert_failed(output)
        self.assertEqual((output / "source_original.json").read_bytes(), source.read_bytes())

    def test_zip_json_limit_is_enforced(self):
        source = self.root / "export.zip"
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("conversations.json", json.dumps(fixture()))
        output = self.root / "imported"
        with self.assertRaisesRegex(ImportValidationError, "json_size_limit"):
            import_history(source, output, max_json_bytes=100)
        self.assert_failed(output)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits only")
    def test_output_permissions_and_umask_restoration(self):
        source = self.source()
        original_mode = stat.S_IMODE(source.stat().st_mode)
        output = self.root / "imported"
        previous = os.umask(0o022)
        try:
            import_history(source, output)
            observed = os.umask(0o022)
            self.assertEqual(observed, 0o022)
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        for path in output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(source.stat().st_mode), original_mode)

    def test_existing_output_is_never_overwritten(self):
        source = self.source()
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "sentinel.txt"
        sentinel.write_text("leave this unchanged")
        with self.assertRaises(FileExistsError):
            import_history(source, output)
        self.assertEqual(sentinel.read_text(), "leave this unchanged")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_unknown_current_path_preserves_every_node(self):
        data = fixture()
        data[0]["current_node"] = "missing-node"
        source = self.source(data)
        output = self.root / "imported"
        manifest = import_history(source, output)
        records = self.records(output)
        self.assertEqual(len(records), 5)
        self.assertTrue(all(row["current_path"] is None for row in records))
        self.assertEqual(manifest["counts"]["conversations_with_unknown_current_path"], 1)

    def test_source_pointer_escapes_opaque_node_id(self):
        data = fixture()
        data[0]["current_node"] = "odd/~node"
        data[0]["mapping"] = {"odd/~node": {"id": "odd/~node", "parent": None, "children": [], "message": None}}
        source = self.source(data)
        output = self.root / "imported"
        import_history(source, output)
        self.assertEqual(self.records(output)[0]["source_ref"]["json_pointer"], "/0/mapping/odd~1~0node")


if __name__ == "__main__":
    unittest.main()

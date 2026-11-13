#!/usr/bin/env python3
"""Import all mapping nodes from a user-provided ChatGPT export, offline.

Test schema, not a guarantee that an account export is complete. Input is one
conversations.json or a ZIP containing exactly one member with that basename.
The complete input file is copied byte-for-byte; ZIP members are never extracted.
The JSONL contains parsed source objects and source pointers, not byte-identical
node slices. The retained source file is authoritative for exact original bytes.
Text is an auxiliary index only. No facts, personality claims, or training labels
are inferred from user questions, assistant messages, or any other content.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


SCHEMA_VERSION = "personal-chat-history-import/0.1.0-test"
DEFAULT_MAX_JSON_BYTES = 256 * 1024 * 1024


class ImportValidationError(ValueError):
    """A sanitized error code; never interpolate conversation contents."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _private_open(path: Path, mode: str = "w"):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.chmod(path, 0o600)
    if "b" in mode:
        return os.fdopen(descriptor, mode)
    return os.fdopen(descriptor, mode, encoding="utf-8", newline="\n")


def _write_manifest(directory: Path, manifest: dict) -> None:
    temporary = directory / "manifest.json.tmp"
    with _private_open(temporary) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / "manifest.json")
    os.chmod(directory / "manifest.json", 0o600)


def _copy_source(source: Path, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, _private_open(destination, "wb") as writer:
        before = os.fstat(reader.fileno())
        while chunk := reader.read(4 * 1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
        after = os.fstat(reader.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
        raise ImportValidationError("source_changed_during_copy")
    return digest.hexdigest(), size


def _read_limited(stream, limit: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = stream.read(min(1024 * 1024, limit + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise ImportValidationError("json_size_limit_exceeded_no_truncation")


def _json_payload(source_copy: Path, is_zip: bool, limit: int) -> tuple[bytes, str | None]:
    if not is_zip:
        if source_copy.stat().st_size > limit:
            raise ImportValidationError("json_size_limit_exceeded_no_truncation")
        with source_copy.open("rb") as stream:
            return _read_limited(stream, limit), None
    with zipfile.ZipFile(source_copy) as archive:
        candidates = [
            info for info in archive.infolist()
            if not info.is_dir()
            and PurePosixPath(info.filename.replace("\\", "/")).name == "conversations.json"
        ]
        if not candidates:
            raise ImportValidationError("zip_has_no_conversations_json")
        if len(candidates) != 1:
            raise ImportValidationError("zip_has_ambiguous_conversations_json_members")
        member = candidates[0]
        if member.file_size > limit:
            raise ImportValidationError("json_size_limit_exceeded_no_truncation")
        if member.flag_bits & 1:
            raise ImportValidationError("encrypted_zip_member_not_supported")
        # Read a member as bytes only. Never use extract/extractall or its name
        # as a local path, including if other entries contain ../ or drive paths.
        with archive.open(member, "r") as stream:
            return _read_limited(stream, limit), member.filename


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImportValidationError("duplicate_json_object_key")
        result[key] = value
    return result


def _reject_nonfinite(_: str):
    raise ImportValidationError("nonfinite_json_number")


def _parse_json(payload: bytes) -> list[dict]:
    try:
        result = json.loads(
            payload.decode("utf-8-sig"), object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ImportValidationError("invalid_or_unsupported_json") from exc
    if not isinstance(result, list):
        raise ImportValidationError("top_level_must_be_conversation_array")
    return result


def _valid_id(value, code: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > 1024 or any(ord(character) < 32 for character in value)):
        raise ImportValidationError(code)
    return value


def _conversation_id(conversation: dict) -> str:
    values = [conversation[key] for key in ("id", "conversation_id")
              if key in conversation and conversation[key] is not None]
    if not values:
        raise ImportValidationError("missing_conversation_id")
    values = [_valid_id(value, "invalid_conversation_id") for value in values]
    if len(set(values)) != 1:
        raise ImportValidationError("conflicting_conversation_ids")
    return values[0]


def _validate_mapping(conversation: dict) -> dict:
    mapping = conversation.get("mapping")
    if not isinstance(mapping, dict):
        raise ImportValidationError("conversation_mapping_must_be_object")
    message_ids = set()
    for key, node in mapping.items():
        _valid_id(key, "invalid_mapping_node_id")
        if not isinstance(node, dict):
            raise ImportValidationError("mapping_node_must_be_object")
        if "id" in node:
            if _valid_id(node["id"], "invalid_node_id") != key:
                raise ImportValidationError("mapping_key_and_node_id_mismatch")
        if node.get("parent") is not None:
            _valid_id(node["parent"], "invalid_parent_id")
        children = node.get("children", [])
        if children is not None:
            if not isinstance(children, list):
                raise ImportValidationError("children_must_be_array_or_null")
            for child in children:
                _valid_id(child, "invalid_child_id")
            if len(set(children)) != len(children):
                raise ImportValidationError("duplicate_child_id")
        message = node.get("message")
        if message is not None:
            if not isinstance(message, dict):
                raise ImportValidationError("message_must_be_object_or_null")
            if message.get("id") is not None:
                message_id = _valid_id(message["id"], "invalid_message_id")
                if message_id in message_ids:
                    raise ImportValidationError("duplicate_message_id_within_conversation")
                message_ids.add(message_id)
    if conversation.get("current_node") is not None:
        _valid_id(conversation["current_node"], "invalid_current_node_id")
    return mapping


def _current_path(conversation: dict, mapping: dict) -> tuple[list[str] | None, str]:
    current = conversation.get("current_node")
    if current is None:
        return None, "current_node_missing"
    path, seen = [], set()
    while current is not None:
        if current in seen:
            return None, "current_path_has_cycle"
        if current not in mapping:
            return None, "current_path_has_missing_node"
        seen.add(current)
        path.append(current)
        current = mapping[current].get("parent")
    return list(reversed(path)), "known"


def _text_index(content) -> str:
    # Do not interpret arbitrary metadata/asset references as conversation text.
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    strings = [part for part in parts if isinstance(part, str)] if isinstance(parts, list) else []
    if isinstance(content.get("text"), str):
        strings.append(content["text"])
    return "\n".join(strings)


def _pointer_key(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def import_history(source: Path | str, output_dir: Path | str,
                   max_json_bytes: int = DEFAULT_MAX_JSON_BYTES) -> dict:
    """Create a new private directory; failed imports keep a failed manifest.

    No existing output is overwritten. Success means complete traversal of the
    provided file's mapping entries, not a verified complete account history.
    """
    if not isinstance(max_json_bytes, int) or isinstance(max_json_bytes, bool) or max_json_bytes <= 0:
        raise ImportValidationError("max_json_bytes_must_be_positive_integer")
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ImportValidationError("input_must_be_regular_file")
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.parent.is_dir():
        raise ImportValidationError("output_parent_directory_must_exist")
    is_zip = source.suffix.lower() == ".zip" or zipfile.is_zipfile(source)
    source_name = "source_original.zip" if is_zip else "source_original.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "importing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coverage_claim": "not_full_account_export_verification",
        "not_full_account_export_verification": True,
        "scope": "All mapping nodes, including non-current branches, in the provided input file only.",
        "source": {"provided_path": str(source), "original_filename": source.name,
                   "copy": source_name},
        "limits": {"max_json_bytes": max_json_bytes, "silent_truncation": False},
        "interpretation": {
            "text_index_is_auxiliary": True,
            "raw_node_is_parsed_json": True,
            "exact_bytes_authority": source_name,
            "user_facts_inferred": False,
            "personality_inferred": False,
            "training_labels_generated": False,
            "assistant_output_is_not_automatically_user_fact": True,
            "current_path": "true/false for a known path, null when path cannot be reconstructed",
            "attachments": "Full source ZIP is retained if supplied; attachment members are not extracted or decoded.",
        },
    }
    previous_umask = os.umask(0o077)
    created_output = False
    try:
        output_dir.mkdir(mode=0o700, exist_ok=False)
        created_output = True
        os.chmod(output_dir, 0o700)
        _write_manifest(output_dir, manifest)
        source_copy = output_dir / source_name
        source_sha, source_size = _copy_source(source, source_copy)
        manifest["source"].update({"sha256": source_sha, "bytes": source_size})
        payload, member = _json_payload(source_copy, is_zip, max_json_bytes)
        manifest["source"].update({"json_archive_member": member, "json_bytes": len(payload),
                                   "json_sha256": hashlib.sha256(payload).hexdigest()})
        conversations = _parse_json(payload)
        del payload
        seen_conversations = set()
        prepared = []
        for conversation in conversations:
            if not isinstance(conversation, dict):
                raise ImportValidationError("conversation_must_be_object")
            conversation_id = _conversation_id(conversation)
            if conversation_id in seen_conversations:
                raise ImportValidationError("duplicate_conversation_id")
            seen_conversations.add(conversation_id)
            mapping = _validate_mapping(conversation)
            path, path_status = _current_path(conversation, mapping)
            prepared.append((conversation_id, conversation, mapping, path, path_status))

        counts = Counter(conversations=len(prepared))
        expected_nodes = sum(len(entry[2]) for entry in prepared)
        conversation_index = []
        with _private_open(output_dir / "messages.jsonl") as stream:
            for index, (conversation_id, conversation, mapping, path, path_status) in enumerate(prepared):
                path_nodes = set(path) if path is not None else None
                if path_nodes is None:
                    counts["conversations_with_unknown_current_path"] += 1
                conversation_index.append({
                    "conversation_id": conversation_id, "source_index": index,
                    "mapping_node_count": len(mapping), "current_path_status": path_status,
                    "current_path_node_ids": path,
                })
                for node_id, node in mapping.items():
                    counts["mapping_nodes"] += 1
                    message = node.get("message")
                    author = message.get("author") if message is not None else None
                    content = message.get("content") if message is not None else None
                    role = author.get("role") if isinstance(author, dict) else None
                    on_path = node_id in path_nodes if path_nodes is not None else None
                    if message is None:
                        counts["nodes_without_message"] += 1
                    else:
                        counts["message_nodes"] += 1
                        if message.get("create_time") is None:
                            counts["messages_without_create_time"] += 1
                        if message.get("id") is None:
                            counts["messages_without_id"] += 1
                        if on_path is False:
                            counts["non_current_branch_message_nodes"] += 1
                    parent = node.get("parent")
                    if parent is not None and parent not in mapping:
                        counts["dangling_parent_references"] += 1
                    for child in node.get("children") or []:
                        if child not in mapping:
                            counts["dangling_child_references"] += 1
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "message" if message is not None else "mapping_node_without_message",
                        "conversation_id": conversation_id,
                        "conversation_title": conversation.get("title"),
                        "node_id": node_id,
                        "parent": parent, "children": node.get("children"),
                        "current_path": on_path, "current_path_status": path_status,
                        "message_id": message.get("id") if message is not None else None,
                        "role": role, "author": author,
                        "timestamps": {
                            "conversation_create_time": conversation.get("create_time"),
                            "conversation_update_time": conversation.get("update_time"),
                            "message_create_time": message.get("create_time") if message is not None else None,
                            "message_update_time": message.get("update_time") if message is not None else None,
                        },
                        "content": content,
                        "metadata": message.get("metadata") if message is not None else None,
                        "text_index": _text_index(content),
                        "interpretation_status": "raw_utterance_no_fact_or_training_label_inferred",
                        "raw_node": node,
                        "source_ref": {
                            "source_file": source_name, "source_sha256": source_sha,
                            "archive_member": member,
                            "json_pointer": f"/{index}/mapping/{_pointer_key(node_id)}",
                        },
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if counts["mapping_nodes"] != expected_nodes:
            raise ImportValidationError("mapping_node_count_mismatch")
        for key in ("mapping_nodes", "message_nodes", "nodes_without_message",
                    "non_current_branch_message_nodes", "messages_without_create_time",
                    "messages_without_id", "conversations_with_unknown_current_path",
                    "dangling_parent_references", "dangling_child_references"):
            counts.setdefault(key, 0)
        manifest.update({
            "status": "complete_for_provided_file",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "counts": dict(counts), "expected_mapping_nodes": expected_nodes,
            "conversation_index": conversation_index,
            "outputs": {"messages": "messages.jsonl", "manifest": "manifest.json", "source": source_name},
        })
        _write_manifest(output_dir, manifest)
        return manifest
    except BaseException as exc:
        if created_output:
            manifest["status"] = "failed"
            manifest["error_type"] = type(exc).__name__
            manifest["error_code"] = exc.code if isinstance(exc, ImportValidationError) else "import_interrupted_or_io_error"
            # Do not include exception text: it could expose user content.
            try:
                pending = output_dir / "manifest.json.tmp"
                if pending.exists():
                    pending.unlink()
                _write_manifest(output_dir, manifest)
            except OSError:
                pass  # The CLI still reports failure if even a manifest cannot be saved.
        raise
    finally:
        os.umask(previous_umask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, dest="source")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-json-bytes", type=int, default=DEFAULT_MAX_JSON_BYTES)
    args = parser.parse_args()
    try:
        manifest = import_history(args.source, args.output_dir, args.max_json_bytes)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        code = exc.code if isinstance(exc, ImportValidationError) else type(exc).__name__
        print(json.dumps({"status": "failed", "error_code": code,
                          "complete_import_reported": False}), file=sys.stderr)
        raise SystemExit(1)
    # Only paths, digests, counts and status; never print message text or titles.
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir.resolve()),
                      "source_sha256": manifest["source"]["sha256"], "counts": manifest["counts"],
                      "coverage_claim": manifest["coverage_claim"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

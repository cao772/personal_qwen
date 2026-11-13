#!/usr/bin/env python3
"""Offline TraceMemo WeChat JSONL importer; preserves source, never makes SFT.

Accepts a JSONL file or a ZIP with exactly one messages.jsonl member. Every row
is streamed and validated; blanks, ambiguous archives, duplicates, and limits
cause an explicit failed import rather than silent skipping/truncation.
The full provided file is retained byte-for-byte. Parsed raw_record objects and
line references preserve provenance; exact original bytes live in the copy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

from import_chat_history import (
    ImportValidationError, _copy_source, _private_open, _reject_nonfinite,
    _unique_object, _write_manifest,
)


SCHEMA_VERSION = "personal-tracememo-import/0.1.0-test"
DEFAULT_MAX_JSONL_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_LINE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_RECORDS = 1_000_000


def _parse_row(raw_line: bytes, first: bool) -> dict:
    if not raw_line.strip():
        raise ImportValidationError("blank_jsonl_line_no_silent_skip")
    try:
        row = json.loads(
            raw_line.decode("utf-8-sig" if first else "utf-8"),
            object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ImportValidationError("invalid_jsonl_record") from exc
    if not isinstance(row, dict):
        raise ImportValidationError("jsonl_record_must_be_object")
    return row


def _validate_row(row: dict) -> datetime:
    for field in ("chat_id", "chat_name", "chat_type", "sender", "sender_id",
                  "time", "server_id", "type", "text"):
        if field not in row or not isinstance(row[field], str):
            raise ImportValidationError(f"missing_or_invalid_{field}_string")
    if not row["chat_id"].strip():
        raise ImportValidationError("empty_chat_id")
    if row["chat_type"] not in ("group", "user"):
        raise ImportValidationError("unsupported_chat_type")
    if not row["type"].strip():
        raise ImportValidationError("empty_message_type")
    if type(row.get("is_self")) is not bool:
        raise ImportValidationError("is_self_must_be_boolean")
    if type(row.get("local_id")) is not int or row["local_id"] < 0:
        raise ImportValidationError("local_id_must_be_nonnegative_integer")
    if "quoted_text" in row and not isinstance(row["quoted_text"], str):
        raise ImportValidationError("quoted_text_must_be_string_when_present")
    try:
        occurred = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ImportValidationError("invalid_iso_timestamp") from exc
    if occurred.tzinfo is None or occurred.utcoffset() is None:
        raise ImportValidationError("timestamp_requires_explicit_timezone")
    try:
        return occurred.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ImportValidationError("timestamp_outside_supported_utc_range") from exc


def _open_jsonl(stack: ExitStack, source_copy: Path, is_zip: bool, limit: int):
    if not is_zip:
        if source_copy.stat().st_size > limit:
            raise ImportValidationError("jsonl_size_limit_exceeded_no_truncation")
        return stack.enter_context(source_copy.open("rb")), None
    archive = stack.enter_context(zipfile.ZipFile(source_copy))
    candidates = [
        info for info in archive.infolist()
        if not info.is_dir()
        and PurePosixPath(info.filename.replace("\\", "/")).name == "messages.jsonl"
    ]
    if not candidates:
        raise ImportValidationError("zip_has_no_messages_jsonl")
    if len(candidates) != 1:
        raise ImportValidationError("zip_has_ambiguous_messages_jsonl_members")
    member = candidates[0]
    if member.file_size > limit:
        raise ImportValidationError("jsonl_size_limit_exceeded_no_truncation")
    if member.flag_bits & 1:
        raise ImportValidationError("encrypted_zip_member_not_supported")
    # The member name is never used as a filesystem path.
    return stack.enter_context(archive.open(member, "r")), member.filename


def import_tracememo(source: Path | str, output_dir: Path | str,
                     max_jsonl_bytes: int = DEFAULT_MAX_JSONL_BYTES,
                     max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
                     max_records: int = DEFAULT_MAX_RECORDS) -> dict:
    for name, value in (("max_jsonl_bytes", max_jsonl_bytes),
                        ("max_line_bytes", max_line_bytes), ("max_records", max_records)):
        if type(value) is not int or value <= 0:
            raise ImportValidationError(f"{name}_must_be_positive_integer")
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ImportValidationError("input_must_be_regular_file")
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.parent.is_dir():
        raise ImportValidationError("output_parent_directory_must_exist")
    is_zip = source.suffix.lower() == ".zip" or zipfile.is_zipfile(source)
    source_name = "source_original.zip" if is_zip else "source_original.jsonl"
    counts = Counter(rows=0, self_rows=0, other_rows=0, quoted_rows=0,
                     blank_text_rows=0, blank_sender_display_name_rows=0)
    message_types = Counter()
    chats, identities = set(), set()
    minimum_time = maximum_time = None

    def statistics() -> dict:
        return {
            **dict(counts), "chats": len(chats), "message_types": dict(message_types),
            "time_min_utc": minimum_time.isoformat() if minimum_time else None,
            "time_max_utc": maximum_time.isoformat() if maximum_time else None,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "importing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coverage_claim": "not_full_account_export_verification",
        "not_full_account_export_verification": True,
        "scope": "Every validated line of the provided messages.jsonl only; not all WeChat/account history.",
        "source": {"provided_path": str(source), "original_filename": source.name, "copy": source_name},
        "limits": {"max_jsonl_bytes": max_jsonl_bytes, "max_line_bytes": max_line_bytes,
                   "max_records": max_records, "silent_truncation": False},
        "interpretation": {
            "speaker_roles": "self/other, solely from the source is_self boolean",
            "other_is_not_assistant": True,
            "text_index_is_auxiliary": True,
            "quoted_text_is_not_parsed_or_attributed_to_owner": True,
            "raw_record_is_parsed_json": True,
            "exact_bytes_authority": source_name,
            "facts_inferred": False, "personality_inferred": False,
            "training_labels_generated": False, "is_sft_dataset": False,
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
        jsonl_digest = hashlib.sha256()
        total_bytes = 0
        with ExitStack() as stack:
            reader, member = _open_jsonl(stack, source_copy, is_zip, max_jsonl_bytes)
            manifest["source"]["jsonl_archive_member"] = member
            writer = stack.enter_context(_private_open(output_dir / "records.jsonl"))
            line_number = 0
            while True:
                raw_line = reader.readline(max_line_bytes + 1)
                if not raw_line:
                    break
                line_number += 1
                total_bytes += len(raw_line)
                if len(raw_line) > max_line_bytes:
                    raise ImportValidationError("line_size_limit_exceeded_no_truncation")
                if total_bytes > max_jsonl_bytes:
                    raise ImportValidationError("jsonl_size_limit_exceeded_no_truncation")
                if counts["rows"] >= max_records:
                    raise ImportValidationError("record_limit_exceeded_no_truncation")
                row = _parse_row(raw_line, first=line_number == 1)
                occurred = _validate_row(row)
                identity = (row["chat_id"], row["local_id"])
                if identity in identities:
                    raise ImportValidationError("duplicate_chat_id_local_id_no_silent_deduplication")
                identities.add(identity)
                chats.add(row["chat_id"])
                speaker_role = "self" if row["is_self"] else "other"
                has_quote = row["type"] == "引用消息" or "quoted_text" in row
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "wechat_message",
                    "record_id": hashlib.sha256(json.dumps(
                        [source_sha, row["chat_id"], row["local_id"]], ensure_ascii=False,
                    ).encode("utf-8")).hexdigest(),
                    "speaker_role": speaker_role, "is_self": row["is_self"],
                    "author": {"id": row["sender_id"], "display_name": row["sender"]},
                    "conversation": {"id": row["chat_id"], "name": row["chat_name"], "type": row["chat_type"]},
                    "time_original": row["time"], "time_utc": occurred.isoformat(),
                    "message_ids": {"local_id": row["local_id"], "server_id": row["server_id"]},
                    "message_type": row["type"], "text_index": row["text"],
                    "quoted_text": row.get("quoted_text"), "has_quote": has_quote,
                    "quote_interpretation": "unparsed_quotation_not_owner_fact" if has_quote else None,
                    "interpretation_status": "raw_utterance_no_fact_or_training_label_inferred",
                    "raw_record": row,
                    "source_ref": {"source_file": source_name, "source_sha256": source_sha,
                                   "archive_member": member, "line": line_number,
                                   "line_sha256": hashlib.sha256(raw_line).hexdigest()},
                }
                writer.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                counts["rows"] += 1
                counts[f"{speaker_role}_rows"] += 1
                counts["quoted_rows"] += int(has_quote)
                counts["blank_text_rows"] += int(not row["text"].strip())
                counts["blank_sender_display_name_rows"] += int(not row["sender"].strip())
                message_types[row["type"]] += 1
                minimum_time = occurred if minimum_time is None else min(minimum_time, occurred)
                maximum_time = occurred if maximum_time is None else max(maximum_time, occurred)
                jsonl_digest.update(raw_line)
            if not counts["rows"]:
                raise ImportValidationError("empty_messages_jsonl")
            writer.flush()
            os.fsync(writer.fileno())
        manifest["source"].update({"jsonl_bytes": total_bytes, "jsonl_sha256": jsonl_digest.hexdigest()})
        manifest.update({"status": "complete_for_provided_file", "counts": statistics(),
                         "completed_at": datetime.now(timezone.utc).isoformat(),
                         "outputs": {"records": "records.jsonl", "manifest": "manifest.json", "source": source_name}})
        _write_manifest(output_dir, manifest)
        return manifest
    except BaseException as exc:
        if created_output:
            manifest.update({"status": "failed", "counts_processed_before_failure": statistics(),
                             "error_type": type(exc).__name__,
                             "error_code": exc.code if isinstance(exc, ImportValidationError) else "import_interrupted_or_io_error"})
            try:
                pending = output_dir / "manifest.json.tmp"
                if pending.exists():
                    pending.unlink()
                _write_manifest(output_dir, manifest)
            except OSError:
                pass
        raise
    finally:
        os.umask(previous_umask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-jsonl-bytes", type=int, default=DEFAULT_MAX_JSONL_BYTES)
    parser.add_argument("--max-line-bytes", type=int, default=DEFAULT_MAX_LINE_BYTES)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    args = parser.parse_args()
    try:
        manifest = import_tracememo(args.source, args.output_dir, args.max_jsonl_bytes,
                                    args.max_line_bytes, args.max_records)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        code = exc.code if isinstance(exc, ImportValidationError) else type(exc).__name__
        print(json.dumps({"status": "failed", "error_code": code, "complete_import_reported": False}), file=sys.stderr)
        raise SystemExit(1)
    # Avoid printing messages, quotes, identities, chat titles, or date ranges.
    public_counts = {key: manifest["counts"][key] for key in (
        "rows", "chats", "self_rows", "other_rows", "quoted_rows",
        "blank_text_rows", "blank_sender_display_name_rows",
    )}
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir.resolve()),
                      "source_sha256": manifest["source"]["sha256"], "counts": public_counts,
                      "coverage_claim": manifest["coverage_claim"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Prepare private reader comparisons from fixed retrieved source lines.

The retriever, question and model prompt stay unchanged. Only evidence formatting
changes. Reference answers and gold labels are never inputs to the packer.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def aware_timestamp(value):
    when = datetime.fromisoformat(value)
    if when.utcoffset() is None:
        raise ValueError("source and query timestamps require UTC offsets")
    return when.timestamp()


def unique_rows(rows):
    result = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("query and retrieval IDs must be unique nonempty strings")
        result[key] = row
    return result


def source_lines(row):
    lines = row["retrieved_source_lines"]
    if (not isinstance(lines, list) or len(lines) > 10
            or any(type(line) is not int or line < 1 for line in lines)
            or len(set(lines)) != len(lines)):
        raise ValueError("retrieval must contain at most ten distinct positive source lines")
    return lines


def selected_messages(archive, lines, expected_sha256):
    if (len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("expected archive SHA256 must be a lowercase digest")
    if digest(archive) != expected_sha256:
        raise ValueError("source archive does not match the frozen source identity")
    required, messages = set(lines), {}
    with zipfile.ZipFile(archive) as container:
        candidates = [member for member in container.infolist() if not member.is_dir()
                      and PurePosixPath(member.filename.replace("\\", "/")).name == "messages.jsonl"]
        if len(candidates) != 1:
            raise ValueError("archive requires exactly one messages.jsonl member")
        if candidates[0].file_size > 1024 * 1024 * 1024:
            raise ValueError("message member exceeds the supported size")
        if not required:
            return messages
        with container.open(candidates[0]) as source:
            number = 0
            while raw := source.readline(8 * 1024 * 1024 + 1):
                number += 1
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("source line exceeds the supported size")
                if number not in required:
                    continue
                item = json.loads(raw)
                if type(item.get("is_self")) is not bool:
                    raise ValueError("source speaker role must be a boolean, not inferred")
                body, quoted = item.get("text", ""), item.get("quoted_text", "")
                if not isinstance(body, str) or not isinstance(quoted, str):
                    raise ValueError("source body and quoted_text must be strings")
                aware_timestamp(item["time"])
                messages[number] = dict(
                    line=number, ts=item["time"], is_self=item["is_self"],
                    body=body, quoted=quoted, chat_id=item["chat_id"],
                    local_id=str(item["local_id"]), sender=item.get("sender", ""),
                    sender_id=item.get("sender_id", ""), chat_name=item.get("chat_name", ""),
                )
                if len(messages) == len(required):
                    break
    if set(messages) != required:
        raise ValueError("retrieval refers to a source line absent from the archive")
    return messages


def legacy_evidence(messages):
    """Reproduce the fixed-prefix format used by the earlier reader experiment."""
    evidence = []
    for message in messages:
        body = " ".join(message["body"].split())[:120]
        quote = " ".join(message["quoted"].split())[:60]
        role = "本人" if message["is_self"] else "他人"
        content = f"[原文行{message['line']} {message['ts']} {role}] {body}"
        if quote:
            content += f" [引用，不能直接归属发送者：{quote}]"
        evidence.append(content)
    return evidence


def prepare(queries, retrieved, messages, archive_sha256, packer, *, max_chars=None):
    query_map, retrieval_map = unique_rows(queries), unique_rows(retrieved)
    if set(query_map) != set(retrieval_map):
        raise ValueError("query and retrieval ID sets differ")
    if max_chars is not None and (type(max_chars) is not int or max_chars < 0):
        raise ValueError("max_chars must be nonnegative")
    legacy_cases, packed_cases, audits = [], [], []
    for query in queries:
        key = query["id"]
        if not isinstance(query.get("question"), str) or not query["question"].strip():
            raise ValueError("query requires text")
        expected_question_hash = hashlib.sha256(query["question"].encode("utf-8")).hexdigest()
        if retrieval_map[key].get("question_sha256") != expected_question_hash:
            raise ValueError("retrieval question fingerprint is missing or differs from the frozen query")
        when = aware_timestamp(query["as_of"])
        if query.get("source_archive_sha256", archive_sha256) != archive_sha256:
            raise ValueError("query labels refer to a different archive")
        excluded_list = query.get("excluded_lines", [])
        if not isinstance(excluded_list, list) or any(type(v) is not int or v < 1 for v in excluded_list):
            raise ValueError("excluded_lines must be positive source lines")
        excluded = set(excluded_list)
        lines = source_lines(retrieval_map[key])
        selected = []
        for line in lines:
            item = messages[line]
            if line in excluded or aware_timestamp(item["ts"]) > when:
                raise ValueError("retrieval contains excluded or future evidence")
            selected.append(item)
        legacy = legacy_evidence(selected)
        legacy_chars = len("\n".join(legacy))
        budget = legacy_chars if max_chars is None else max_chars
        # No ID, reference, answerable flag, gold line or rubric enters the packer.
        packed = packer(query["question"], selected, archive_sha256=archive_sha256,
                        max_chars=budget, max_messages=10)
        evidence = packed["evidence"]
        rendered = "\n".join(evidence)
        if rendered != packed["rendered_text"] or len(rendered) > budget:
            raise ValueError("packer violated its rendered-text budget contract")
        legacy_cases.append({"id": key, "question": query["question"], "correct": legacy})
        packed_cases.append({"id": key, "question": query["question"], "correct": evidence})
        audits.append({"id": key, "as_of": query["as_of"], "source_lines": lines,
                       "budget_mode": "same_as_legacy" if max_chars is None else "fixed_total_characters",
                       "max_chars": budget, "legacy_chars": legacy_chars,
                       "packed_chars": len(rendered), "pack": packed})
    return legacy_cases, packed_cases, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True,
                        help="frozen retrieval.jsonl with id, question_sha256 and retrieved_source_lines")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-kind", choices=["retrieved", "oracle"], default="retrieved",
                        help="declare labelled-source oracle diagnostics explicitly")
    parser.add_argument("--max-chars", type=int,
                        help="fixed total ceiling; omit to match each legacy rendered character count")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("use a new private output directory")
    queries, retrieved = read_jsonl(args.queries), read_jsonl(args.retrieval)
    if any(row.get("source_kind", args.source_kind) != args.source_kind for row in retrieved):
        raise ValueError("source trace kind differs from the declared comparison")
    wanted = {line for row in retrieved for line in source_lines(row)}
    messages = selected_messages(args.archive, wanted, args.archive_sha256)
    from experiments import evidence_packing
    legacy, packed, audits = prepare(queries, retrieved, messages, args.archive_sha256,
                                      evidence_packing.pack_evidence, max_chars=args.max_chars)
    os.umask(0o077)
    args.out.mkdir(parents=True, mode=0o700, exist_ok=False)
    for filename, rows in [("legacy_cases.jsonl", legacy), ("packed_cases.jsonl", packed),
                           ("packing_audit.jsonl", audits)]:
        (args.out / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                        encoding="utf-8")
    manifest = dict(
        source_archive_sha256=args.archive_sha256, queries_sha256=digest(args.queries),
        retrieval_sha256=digest(args.retrieval), preparer_sha256=digest(__file__),
        packer_sha256=digest(evidence_packing.__file__), queries=len(queries),
        budget_mode="same_as_legacy" if args.max_chars is None else "fixed_total_characters",
        max_chars=args.max_chars,
        source_kind=args.source_kind,
        source_selection=("frozen retrieval; no reranking or new retrieval" if args.source_kind == "retrieved"
                          else "labelled source oracle; reading diagnostic, not retriever performance"),
        model_generation_run=False, answer_accuracy=None,
        limitations=["Unicode characters are not model tokens.",
                     "Span visibility does not demonstrate correct understanding or answer entailment.",
                     "The reader still needs evaluation with the prepared cases."],
        outputs={name: digest(args.out / name) for name in
                 ["legacy_cases.jsonl", "packed_cases.jsonl", "packing_audit.jsonl"]},
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"cases": len(queries), "budget_mode": manifest["budget_mode"],
                      "output": str(args.out), "model_generation_run": False}))


if __name__ == "__main__":
    main()

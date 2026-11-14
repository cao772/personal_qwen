"""Verify frozen synthetic reader-output bindings and prepare a blind review file.

Standard library only. This does not run a model, load a tokenizer, score answers,
or establish that supplied output records came from an actual model execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


SUPPORTED_READER_SHA256 = "f61cd69be33495206257a7afa743dec45c35fbe1051596504e46e1e0dbd7fda2"
HASH_FORMAT = "canonical_json_utf8_v1"
DEFAULT_SHUFFLE_SEED = "synthetic-format-blind-v1"
BUNDLE_FILES = {"cases.jsonl", "condition_map.jsonl", "answer_key.jsonl",
                "contrasts.jsonl", "source_catalog.json"}
OUTPUT_FIELDS = set("""
id condition answer tokens elapsed hit_cap input_tokens evidence_sha256
chat_messages_sha256 input_token_ids_sha256 hash_format requested_prompt_version
effective_prompt_version max_input_tokens max_new_tokens case_file_sha256
""".split())


class AuditError(ValueError):
    """A supplied record does not satisfy the frozen request contract."""


def require(condition, label):
    if not condition:
        raise AuditError(label)


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def fields(value, required, label, optional=()):
    require(type(value) is dict, f"{label}: expected an object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            f"{label}: missing or unknown fields")


def required_fields(value, required, label):
    """Frozen descriptive metadata may contain additional, unused fields."""
    require(type(value) is dict and set(required) <= set(value), f"{label}: missing fields")


def exact(value, expected, label):
    require(canonical_bytes(value) == canonical_bytes(expected), f"{label}: unexpected value")


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum,
            f"{label}: expected integer in [{minimum}, {maximum}]")


def string(value, label, *, empty=False):
    require(type(value) is str and (empty or bool(value)), f"{label}: expected a string")


def strings(value, label):
    require(type(value) is list and all(type(item) is str for item in value),
            f"{label}: expected a list of strings")


def digest(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{label}: expected a lowercase SHA256")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON object contains a duplicate key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise AuditError("JSON contains a non-finite constant")


def parse_json(raw, label):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid UTF-8 JSON") from exc


def parse_jsonl(raw, label):
    # Reject blank records, including trailing extra blank lines. A single final
    # newline is normal and splitlines does not produce an empty record for it.
    lines = raw.splitlines()
    require(bool(lines), f"{label}: no records")
    require(all(line.strip() for line in lines), f"{label}: blank record")
    return [parse_json(line, f"{label} line {i}") for i, line in enumerate(lines, 1)]


def read_file(path, label):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), f"{label}: expected a regular file")
    return path.read_bytes()


def index_rows(rows, label, id_field="id"):
    result = {}
    for row in rows:
        require(type(row) is dict and id_field in row, f"{label}: missing identifier")
        string(row[id_field], f"{label} identifier")
        require(row[id_field] not in result, f"{label}: duplicate identifier")
        result[row[id_field]] = row
    return result


def validate_manifest(manifest):
    required_fields(manifest, {"version", "status", "synthetic", "model_generation_run",
                              "cases", "scenarios", "paired_contrasts", "generation_protocol",
                              "reader_sha256", "files"}, "manifest")
    string(manifest["version"], "manifest.version")
    for name, expected in {
        "status": "prepared_not_executed", "synthetic": True, "model_generation_run": False,
    }.items():
        exact(manifest[name], expected, f"manifest.{name}")
    for name in ("cases", "scenarios", "paired_contrasts"):
        integer(manifest[name], 1, 10**7, f"manifest.{name}")
    protocol = manifest["generation_protocol"]
    expected = {"mode": "research", "prompt_version": "original", "conditions": ["correct"],
                "do_sample": False, "enable_thinking": False, "max_input_tokens": 4096,
                "max_new_tokens": 192, "adapter": None}
    fields(protocol, set(expected) | {"model"}, "manifest.generation_protocol")
    for name, value in expected.items():
        exact(protocol[name], value, f"generation_protocol.{name}")
    string(protocol["model"], "generation_protocol.model")
    exact(manifest["reader_sha256"], SUPPORTED_READER_SHA256, "manifest.reader_sha256")
    required_fields(manifest["files"], BUNDLE_FILES, "manifest.files")
    for name, entry in manifest["files"].items():
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None,
                "manifest.files: expected a simple filename")
        fields(entry, {"sha256", "bytes"}, f"manifest.files.{name}")
        digest(entry["sha256"], f"manifest.files.{name}.sha256")
        integer(entry["bytes"], 1, 10**10, f"manifest.files.{name}.bytes")


def load_bundle(bundle, expected_manifest_sha256):
    digest(expected_manifest_sha256, "expected manifest SHA256")
    bundle = Path(bundle)
    raw_manifest = read_file(bundle / "manifest.json", "manifest")
    exact(sha256(raw_manifest), expected_manifest_sha256, "external manifest binding")
    manifest = parse_json(raw_manifest, "manifest")
    validate_manifest(manifest)
    raw_files = {}
    for name in sorted(manifest["files"]):
        raw = read_file(bundle / name, name)
        exact(len(raw), manifest["files"][name]["bytes"], f"{name} byte length")
        exact(sha256(raw), manifest["files"][name]["sha256"], f"{name} SHA256")
        raw_files[name] = raw
    cases = index_rows(parse_jsonl(raw_files["cases.jsonl"], "cases"), "cases")
    keys = index_rows(parse_jsonl(raw_files["answer_key.jsonl"], "answer key"), "answer key")
    mapping = index_rows(parse_jsonl(raw_files["condition_map.jsonl"], "condition map"), "condition map")
    exact(len(cases), manifest["cases"], "manifest case count")
    require(set(cases) == set(keys) == set(mapping), "bundle identifier sets differ")
    source = parse_json(raw_files["source_catalog.json"], "source catalog")
    required_fields(source, {"synthetic", "scenarios"}, "source catalog")
    exact(source["synthetic"], True, "source catalog synthetic flag")
    require(type(source["scenarios"]) is list, "source scenarios: expected a list")
    scenarios = index_rows(source["scenarios"], "source scenarios", "scenario_id")
    exact(len(scenarios), manifest["scenarios"], "manifest scenario count")
    for scenario in scenarios.values():
        required_fields(scenario, {"scenario_id", "collision_subject", "question", "messages"}, "source scenario")
        string(scenario["question"], "source scenario.question")
        require(scenario["collision_subject"] in ("self", "other"), "invalid collision subject")
        require(type(scenario["messages"]) is list, "source messages: expected a list")
    for case_id, case in cases.items():
        fields(case, {"id", "question", "correct"}, "case")
        string(case["question"], "case.question")
        strings(case["correct"], "case.correct")
        key, meta = keys[case_id], mapping[case_id]
        fields(key, {"id", "scenario_id", "answerable", "expected_answer", "requirements",
                     "forbidden_claims", "grading_rule"}, "answer key")
        require(type(key["answerable"]) is bool, "answerable: expected bool")
        for name in ("expected_answer", "grading_rule"):
            string(key[name], f"answer key.{name}")
        for name in ("requirements", "forbidden_claims"):
            strings(key[name], f"answer key.{name}")
        required_fields(meta, {"id", "scenario_id", "factors", "include_display_name", "collision_subject",
                              "source_messages_sha256", "question_sha256", "evidence_sha256"}, "condition map")
        require(type(meta["factors"]) is dict, "condition map.factors: expected an object")
        require(type(meta["include_display_name"]) is bool, "include_display_name: expected bool")
        string(meta["scenario_id"], "condition map.scenario_id")
        require(meta["scenario_id"] in scenarios, "condition map references unknown scenario")
        exact(key["scenario_id"], meta["scenario_id"], "answer-key scenario binding")
        scenario = scenarios[meta["scenario_id"]]
        exact(meta["collision_subject"], scenario["collision_subject"], "collision subject binding")
        exact(case["question"], scenario["question"], "source question binding")
        for name, actual in {
            "source_messages_sha256": sha256(canonical_bytes(scenario["messages"])),
            "question_sha256": sha256(case["question"].encode("utf-8")),
            "evidence_sha256": sha256(canonical_bytes(case["correct"])),
        }.items():
            exact(meta[name], actual, f"condition map.{name}")
    contrasts = parse_jsonl(raw_files["contrasts.jsonl"], "contrasts")
    exact(len(contrasts), manifest["paired_contrasts"], "manifest contrast count")
    seen = set()
    for contrast in contrasts:
        required_fields(contrast, {"scenario_id", "factor", "left_id", "right_id"}, "contrast")
        for name in ("scenario_id", "factor", "left_id", "right_id"):
            string(contrast[name], f"contrast.{name}")
        signature = tuple(contrast[name] for name in ("factor", "left_id", "right_id"))
        require(signature not in seen, "duplicate contrast")
        seen.add(signature)
        require(contrast["left_id"] in cases and contrast["right_id"] in cases, "contrast references unknown case")
        for side in ("left", "right"):
            exact(mapping[contrast[f"{side}_id"]]["scenario_id"], contrast["scenario_id"], "contrast scenario binding")
    return manifest, cases, keys, mapping


def verified_reader(reader_file, manifest):
    raw = read_file(reader_file, "reader")
    actual = sha256(raw)
    exact(actual, SUPPORTED_READER_SHA256, "reviewed reader source binding")
    exact(actual, manifest["reader_sha256"], "manifest reader binding")
    # Only the exact reviewed source is executable here. Its top level imports
    # stdlib only; torch/transformers and GPU execution remain inside run().
    namespace = {"__name__": "_verified_format_audit_reader", "__file__": str(reader_file)}
    exec(compile(raw, str(reader_file), "exec"), namespace)
    return namespace["chat_messages"]


def validate_answers(raw_answers, cases, manifest, chat_messages):
    answers = index_rows(parse_jsonl(raw_answers, "reader output"), "reader output")
    require(set(answers) == set(cases), "reader output has missing or extra case IDs")
    for case_id, answer in answers.items():
        fields(answer, OUTPUT_FIELDS, "reader output")
        case = cases[case_id]
        messages = chat_messages(case, case["correct"], mode="research", prompt_version="original")
        for name, expected in {
            "condition": "correct", "hash_format": HASH_FORMAT,
            "requested_prompt_version": "original", "effective_prompt_version": "original",
            "max_new_tokens": 192, "max_input_tokens": 4096,
            "case_file_sha256": manifest["files"]["cases.jsonl"]["sha256"],
            "evidence_sha256": sha256(canonical_bytes(case["correct"])),
            "chat_messages_sha256": sha256(canonical_bytes(messages)),
        }.items():
            exact(answer[name], expected, f"reader output.{name}")
        string(answer["answer"], "reader output.answer", empty=True)
        integer(answer["input_tokens"], 1, 4096, "reader output.input_tokens")
        integer(answer["tokens"], 0, 192, "reader output.tokens")
        require(not answer["answer"] or answer["tokens"] > 0,
                "reader output: nonempty answer with zero output tokens")
        exact(answer["hit_cap"], answer["tokens"] == 192, "reader output.hit_cap")
        elapsed = answer["elapsed"]
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                "reader output.elapsed: expected a finite nonnegative number")
        # Counts and this digest are recorded claims, not tokenizer verification.
        digest(answer["input_token_ids_sha256"], "reader output.input_token_ids_sha256")
    return answers


def audit_run(*, bundle, answers_file, reader_file, manifest_sha256, out,
              shuffle_seed=DEFAULT_SHUFFLE_SEED):
    """Validate all records before creating fresh private review artifacts."""
    out = Path(out)
    require(not out.exists() and not out.is_symlink(), "output directory already exists")
    string(shuffle_seed, "shuffle seed")
    manifest, cases, keys, mapping = load_bundle(bundle, manifest_sha256)
    chat_messages = verified_reader(reader_file, manifest)
    raw_answers = read_file(answers_file, "reader output")
    answers = validate_answers(raw_answers, cases, manifest, chat_messages)
    order = sorted(cases, key=lambda case_id: (sha256(canonical_bytes([shuffle_seed, case_id])), case_id))
    blind_rows, blind_mapping = [], []
    for ordinal, case_id in enumerate(order, 1):
        blind_id = f"B{ordinal:04d}"
        case, answer, key = cases[case_id], answers[case_id], keys[case_id]
        blind_rows.append({
            "blind_id": blind_id, "question": case["question"], "evidence": case["correct"],
            "answer": answer["answer"], "answerable": key["answerable"],
            "rubric": {name: key[name] for name in
                       ("expected_answer", "requirements", "forbidden_claims", "grading_rule")},
            "generation": {name: answer[name] for name in ("tokens", "max_new_tokens", "hit_cap")},
        })
        blind_mapping.append({**mapping[case_id], "blind_id": blind_id})
    shuffle = {"algorithm": "sort_sha256_canonical_json_seed_and_case_id_v1", "seed": shuffle_seed}
    blindness = "Original case IDs and condition labels are hidden; evidence formatting, content and record numbers remain visible."
    artifacts = {
        "blind_cases.jsonl": b"".join(canonical_bytes(row) + b"\n" for row in blind_rows),
        "blind_key.json": canonical_bytes({"manifest_sha256": manifest_sha256,
                                           "shuffle": shuffle, "mapping": blind_mapping}) + b"\n",
    }
    input_counts = [row["input_tokens"] for row in answers.values()]
    report = {
        "version": "synthetic-format-run-audit-v1",
        "status": "request_bindings_verified_blind_review_prepared",
        "synthetic_cases": True, "records": len(answers),
        "manifest_sha256": manifest_sha256, "reader_sha256": manifest["reader_sha256"],
        "answers_file_sha256": sha256(raw_answers), "bundle_files": manifest["files"],
        "declared_generation_protocol": manifest["generation_protocol"],
        "input_tokens_reported": {"min": min(input_counts), "max": max(input_counts)},
        "output_cap_records": sum(row["hit_cap"] for row in answers.values()),
        "empty_answer_records": sum(not row["answer"] for row in answers.values()),
        "checks": {"complete_unique_case_ids": True, "strict_reader_output_schema": True,
                   "strict_case_schema": True,
                   "case_file_hash_recomputed": True, "evidence_hashes_recomputed": True,
                   "chat_messages_hashes_recomputed": True,
                   "reported_token_counts_within_caps": True},
        "tokenizer_loaded": False, "input_token_ids_sha256_recomputed": False,
        "input_token_count_recomputed": False, "output_token_count_recomputed": False,
        "model_execution_verified": False, "model_and_adapter_identity_verified": False,
        "run_device_verified": False, "tokenization_verified": False,
        "decoding_execution_verified": False,
        "semantic_scoring_performed": False, "answers_modified": False,
        "shuffle": shuffle, "blinding_limit": blindness,
        "limitations": [
            "This verifies supplied records and request bindings, not the authenticity of model execution.",
            "Token-ID digests are syntax-checked only; actual IDs and tokenizer are unavailable.",
            "Token counts and cap flags are checked for consistency, not recomputed from text.",
            "Model/adapter identity is declared in the plan, not independently recorded by this reader.",
            "Device, weights, environment and actual decoding execution need separate execution-side provenance.",
            "Auxiliary files are hash-checked; packer spans and experimental factor design are not re-audited here.",
            "hit_cap=true is retained for manual review and does not automatically fail or grade an answer.",
            "Give the evaluator only blind_cases.jsonl; keep the key, audit and original bundle separate.",
        ],
        "artifacts": {name: {"sha256": sha256(raw), "bytes": len(raw)}
                      for name, raw in artifacts.items()},
    }
    artifacts["request_audit.json"] = canonical_bytes(report) + b"\n"
    previous_umask = os.umask(0o077)
    try:
        out.mkdir(parents=True, mode=0o700)
        for name, raw in artifacts.items():
            fd = os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
    finally:
        os.umask(previous_umask)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True,
                        help="Externally frozen hash of exact manifest.json bytes")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New private output directory")
    parser.add_argument("--shuffle-seed", default=DEFAULT_SHUFFLE_SEED)
    args = parser.parse_args()
    try:
        result = audit_run(bundle=args.bundle, answers_file=args.answers, reader_file=args.reader,
                           manifest_sha256=args.manifest_sha256, out=args.out,
                           shuffle_seed=args.shuffle_seed)
    except (AuditError, OSError) as exc:
        parser.exit(2, f"audit failed: {exc}\n")
    print(json.dumps({"status": result["status"], "records": result["records"],
                      "output_cap_records": result["output_cap_records"],
                      "semantic_scoring_performed": False}))


if __name__ == "__main__":
    main()

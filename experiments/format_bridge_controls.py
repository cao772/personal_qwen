"""Prepare synthetic bridges to frozen legacy and evidence-packer formats.

The previously frozen factorial generator and its artifacts are not modified.
This script calls the exact pinned public components on the same synthetic facts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from experiments import format_controls as matrix, evidence_packing
import prepare_evidence_cases


VERSION = "synthetic-format-bridge-v10-1"
MAX_EVIDENCE_CHARS = 8192
EXPECTED_COMPONENTS = {
    "matrix_generator": "6d64bc71833c892fb653ad53ae67fa3a8a3f15b33ff68d3d76d6f73cdf0eacf4",
    "legacy_preparer": "dafa8636c70533d7693617c0cdfdcd1d2e7585469f35dfffd161f4fe74854f81",
    "evidence_packer": "961f83c7ad3836c90401bdfaccf33c387488f25f30b9fd242ef1ff2495919ba3",
}
EXPECTED_READER = "f61cd69be33495206257a7afa743dec45c35fbe1051596504e46e1e0dbd7fda2"
EXPECTED_MATRIX_CASES = "42ca58d89f5b0db26e2a3d1d60f55bc745ee657aec15881081e2496b2b1e2ac8"
GROUPS = {
    "A": {"presentation": "legacy_prefix", "display_name_collision": "neutral"},
    "B": {"presentation": "frozen_packer_full", "display_name_collision": "neutral"},
    "C": {"presentation": "frozen_packer_full", "display_name_collision": "assistant"},
}


def _digest(path):
    return matrix.sha256(Path(path).read_bytes())


def load_components(legacy_preparer, packer_file, reader_file):
    if _digest(matrix.__file__) != EXPECTED_COMPONENTS["matrix_generator"]:
        raise ValueError("the original synthetic matrix generator changed")
    if _digest(reader_file) != EXPECTED_READER:
        raise ValueError("reader source differs from the frozen identity")
    for name, module, path in (("legacy_preparer", prepare_evidence_cases, legacy_preparer),
                               ("evidence_packer", evidence_packing, packer_file)):
        if (_digest(module.__file__) != EXPECTED_COMPONENTS[name]
                or _digest(path) != EXPECTED_COMPONENTS[name]):
            raise ValueError("component source differs from the frozen identity: " + name)
    return prepare_evidence_cases, evidence_packing


def _identity(scenario_id, group):
    return "brg-" + matrix.sha256(matrix.canonical_bytes([VERSION, scenario_id, group]))[:20]


def source_messages(scenario, index, collision_level):
    """Adapt unchanged synthetic records to the existing public component schema."""
    messages = []
    for source in scenario["messages"]:
        messages.append({
            "line": (index + 1) * 100 + source["number"],
            "ts": source["timestamp"], "is_self": source["is_self"],
            "body": source["body"], "quoted": source["quoted"],
            "sender": matrix._name(source, scenario["collision_subject"], collision_level),
            "sender_id": "synthetic-self" if source["is_self"] else "synthetic-other",
            "chat_id": "synthetic-" + scenario["scenario_id"], "chat_name": "合成场景",
        })
    return messages


def validate_legacy_sources(messages):
    for message in messages:
        if type(message["is_self"]) is not bool:
            raise ValueError("synthetic source role must be boolean")
        for field, limit in (("body", 120), ("quoted", 60)):
            value = message[field]
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError("legacy prefix would clip a synthetic source field")
            if " ".join(value.split()) != value:
                raise ValueError("legacy whitespace normalization would alter a synthetic source field")


def validate_complete_pack(pack, messages, source_hash):
    if (pack["archive_sha256"] if "archive_sha256" in pack else source_hash) != source_hash:
        raise ValueError("packer changed the synthetic source identity")
    if (pack["input_messages"] != len(messages) or pack["supplied_messages"] != len(messages)
            or len(pack["records"]) != len(messages) or len(pack["evidence"]) != len(messages) + 1
            or pack["empty_reason"] is not None):
        raise ValueError("bridge requires every complete source message")
    if pack["rendered_text"] != "\n".join(pack["evidence"]):
        raise ValueError("rendered evidence differs from the reader input")
    if (pack["budget"]["maximum"] != MAX_EVIDENCE_CHARS
            or pack["budget"]["used"] != len(pack["rendered_text"])
            or len(pack["rendered_text"]) > MAX_EVIDENCE_CHARS):
        raise ValueError("packer budget audit differs from actual input")
    for original, record, rendered in zip(messages, pack["records"], pack["evidence"][1:]):
        if not record["included"] or record["omission_reason"] is not None:
            raise ValueError("a bridge source was omitted")
        for key in ("line", "ts", "is_self", "sender", "sender_id", "chat_id", "chat_name"):
            if record[key] != original[key]:
                raise ValueError("packer changed source metadata")
        if record["archive_sha256"] != source_hash:
            raise ValueError("a source record has the wrong synthetic identity")
        role = "本人" if original["is_self"] else "他人"
        header = (f"[消息#{original['line']}] " + json.dumps(original["ts"], ensure_ascii=False)
                  + " " + role + " " + json.dumps(original["sender"], ensure_ascii=False))
        expected_lines = [header]
        for field in ("body", "quoted"):
            value = original[field]
            audit = record["fields"][field]
            spans = [{"start": 0, "end": len(value), "text": value}] if value else []
            if (audit["spans"] != spans or audit["omitted_ranges"] or audit["complete"] is not True
                    or audit["source_length"] != len(value)
                    or audit["source_text_sha256"] != matrix.sha256(value.encode())):
                raise ValueError("bridge source field is not presented in full")
            if value:
                label = "正文" if field == "body" else "引用（原作者未核实）"
                expected_lines.append(label + "：" + json.dumps(value, ensure_ascii=False))
            elif field == "body":
                expected_lines.append("正文为空。")
        if rendered != "\n".join(expected_lines):
            raise ValueError("model text differs from the frozen full-field representation")


def build_bridge(legacy, packer):
    frozen_matrix = matrix.build_experiment()
    if matrix.sha256(matrix._jsonl(frozen_matrix["cases"])) != EXPECTED_MATRIX_CASES:
        raise ValueError("synthetic source questions changed since the matrix was frozen")
    catalog = deepcopy(frozen_matrix["source_catalog"])
    catalog_bytes = matrix.canonical_bytes({"synthetic": True, "scenarios": catalog}) + b"\n"
    source_hash = matrix.sha256(catalog_bytes)
    scenarios = matrix.synthetic_scenarios()
    cases, maps, keys, contrasts, audits = [], [], [], [], []
    for index, scenario in enumerate(scenarios):
        neutral = source_messages(scenario, index, "neutral")
        collision = source_messages(scenario, index, "assistant")
        validate_legacy_sources(neutral)
        legacy_input = legacy.legacy_evidence(neutral)
        packs = {}
        for group, messages in (("B", neutral), ("C", collision)):
            packs[group] = packer.pack_evidence(scenario["question"], messages,
                                               archive_sha256=source_hash,
                                               max_chars=MAX_EVIDENCE_CHARS, max_messages=10)
            validate_complete_pack(packs[group], messages, source_hash)
        # The name intervention changes only the selected subject's sender field.
        for a, b in zip(neutral, collision):
            if {k: v for k, v in a.items() if k != "sender"} != {k: v for k, v in b.items() if k != "sender"}:
                raise ValueError("name intervention altered source semantics")
        for group, factors in GROUPS.items():
            evidence = legacy_input if group == "A" else packs[group]["evidence"]
            identifier = _identity(scenario["scenario_id"], group)
            cases.append({"id": identifier, "question": scenario["question"], "correct": evidence})
            maps.append({
                "id": identifier, "scenario_id": scenario["scenario_id"], "bridge_group": group,
                "factors": dict(factors), "include_display_name": group != "A",
                "collision_subject": scenario["collision_subject"],
                "source_messages_sha256": matrix.sha256(matrix.canonical_bytes(scenario["messages"])),
                "question_sha256": matrix.sha256(scenario["question"].encode()),
                "evidence_sha256": matrix.sha256(matrix.canonical_bytes(evidence)),
            })
            keys.append({"id": identifier, "scenario_id": scenario["scenario_id"],
                         **{name: deepcopy(scenario[name]) for name in
                            ("answerable", "expected_answer", "requirements", "forbidden_claims")},
                         "grading_rule": "All required facts and subject assignments must hold; no forbidden claim. Equivalent natural Chinese is accepted. Do not require verbatim wording."})
            if group != "A":
                audits.append({"id": identifier, "scenario_id": scenario["scenario_id"],
                               "bridge_group": group, "pack": packs[group]})
        contrasts.extend([
            {"scenario_id": scenario["scenario_id"], "factor": "presentation_bundle",
             "left_id": _identity(scenario["scenario_id"], "A"),
             "right_id": _identity(scenario["scenario_id"], "B"),
             "left_level": "legacy_prefix", "right_level": "frozen_packer_full",
             "fixed_factors": {"display_name_collision": "neutral"},
             "causal_interpretation": "composite_format_bridge"},
            {"scenario_id": scenario["scenario_id"], "factor": "display_name_collision",
             "left_id": _identity(scenario["scenario_id"], "B"),
             "right_id": _identity(scenario["scenario_id"], "C"),
             "left_level": "neutral", "right_level": "assistant",
             "fixed_factors": {"presentation": "frozen_packer_full"},
             "causal_interpretation": "single_display_name_factor"},
        ])
    return {
        "cases": sorted(cases, key=lambda row: row["id"]),
        "condition_map": sorted(maps, key=lambda row: row["id"]),
        "answer_key": sorted(keys, key=lambda row: row["id"]),
        "contrasts": sorted(contrasts, key=lambda row: (row["factor"], row["scenario_id"])),
        "source_catalog": catalog, "source_catalog_bytes": catalog_bytes,
        "pack_audit": sorted(audits, key=lambda row: row["id"]),
        "synthetic_source_sha256": source_hash,
    }


def write_bridge(output, *, legacy_preparer, packer_file, reader_file):
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new bridge output directory")
    legacy, packer = load_components(legacy_preparer, packer_file, reader_file)
    result = build_bridge(legacy, packer)
    files = {name + ".jsonl": matrix._jsonl(result[name]) for name in
             ("cases", "condition_map", "answer_key", "contrasts", "pack_audit")}
    files["source_catalog.json"] = result["source_catalog_bytes"]
    sizes = [len("\n".join(row["correct"])) for row in result["cases"]]
    manifest = {
        "version": VERSION, "design_kind": "bridge_not_factorial", "status": "prepared_not_executed",
        "synthetic": True, "model_generation_run": False, "model_answers": 0, "model_accuracy": None,
        "actual_input_tokens": None, "training_run": False, "personalization_validated": False,
        "scenarios": 8, "bridge_conditions_per_scenario": 3, "cases": 24, "paired_contrasts": 16,
        "factor_levels": {"presentation": ["legacy_prefix", "frozen_packer_full"],
                          "display_name_collision": ["neutral", "assistant"]},
        "unchanged_fields": ["question", "message order", "source timestamp", "source role", "body", "quoted", "reference answer", "grading rubric"],
        "collision_subject_balance": {"self": 4, "other": 4},
        "evidence_unicode_characters": {"min": min(sizes), "max": max(sizes)},
        "generation_protocol": {"mode": "research", "prompt_version": "original", "conditions": ["correct"],
                                "do_sample": False, "enable_thinking": False,
                                "max_input_tokens": 4096, "max_new_tokens": 192,
                                "adapter": None, "model": "same pinned existing local baseline; verify at execution"},
        "generator_sha256": _digest(__file__), "reader_sha256": EXPECTED_READER,
        "external_components_sha256": dict(EXPECTED_COMPONENTS),
        "matrix_cases_sha256": EXPECTED_MATRIX_CASES,
        "synthetic_source_sha256": result["synthetic_source_sha256"],
        "max_evidence_chars": MAX_EVIDENCE_CHARS,
        "files": {name: {"sha256": matrix.sha256(value), "bytes": len(value)} for name, value in files.items()},
        "analysis_plan": [
            "Report all eight A/B composite-format bridges and all eight B/C display-name pairs; no selective pair reporting.",
            "Keep the 136-case controlled matrix and this 24-case bridge separate in condition-level reporting.",
            "Compare substantive answer correctness, subject errors, unsupported claims and metadata-only outputs using the same frozen scenario rubrics.",
            "A/B changes several presentation elements together and cannot identify a single causal formatting feature.",
            "B/C changes only the predesignated subject's displayed name; source role, fact content, complete pack and prompt remain fixed.",
        ],
        "limitations": [
            "These are the same eight synthetic scenarios as the controlled matrix, not eight new independent tasks or personal-history acceptance cases.",
            "The combined 136+24=160 inputs remain dependent variants of eight researcher-authored scenarios.",
            "The original frozen legacy_evidence and pack_evidence implementations are called directly. Imported modules and declared source paths are hash-checked before invoking the formatters.",
            "Legacy fields are shorter than the 120/60 prefix limits and unchanged by whitespace normalization; every packer body/quote is verified complete.",
            "The packer API archive_sha256 is bound to this synthetic source catalog, not to a real chat ZIP.",
            "Self/other collision groups use different scenarios; group summaries are descriptive and do not establish which subject is more vulnerable.",
            "Display-name substitution changes string identity and tokenization together; no equal-token or model-internal mechanism claim is made.",
            "Absence of an error on these synthetic bridges cannot prove that it is absent from earlier real inputs.",
            "No tokenizer or model has run. Input-token counts and answer scores are unknown until actual execution.",
        ],
    }
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    (output / "manifest.json").write_bytes(matrix.canonical_bytes(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-preparer", type=Path, required=True)
    parser.add_argument("--packer-file", type=Path, required=True)
    parser.add_argument("--reader-file", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_bridge(args.out, legacy_preparer=args.legacy_preparer,
                            packer_file=args.packer_file, reader_file=args.reader_file)
    print(json.dumps({"cases": manifest["cases"], "paired_contrasts": manifest["paired_contrasts"],
                      "status": manifest["status"], "output": str(args.out)}))


if __name__ == "__main__":
    main()

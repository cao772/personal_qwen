"""Prepare synthetic, paired author-metadata and serialization diagnostics.

No model, tokenizer, user history, network or training is used. Generated reader
inputs contain only id/question/correct. Answer keys and factor maps are separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path


VERSION = "synthetic-format-controls-v10-1"
FACTORS = {
    "display_name_collision": ("neutral", "assistant"),
    "string_serialization": ("plain", "json"),
    "author_block_position": ("before_body", "after_body"),
    "extra_is_self_field": (False, True),
}
DEFAULT_FACTORS = {name: levels[0] for name, levels in FACTORS.items()}


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def _message(number, is_self, body, quoted=""):
    return {"number": number, "timestamp": f"2025-03-10T09:{number * 10:02d}:00+08:00",
            "is_self": is_self, "body": body, "quoted": quoted}


def synthetic_scenarios():
    """All text below was invented for this diagnostic, not copied from history."""
    return [
        {
            "scenario_id": "S01", "domain": "daily_life", "skills": ["explicit_negation"],
            "collision_subject": "self",
            "question": "这段模拟记录中，我周二晚上有没有去健身房，实际做了什么？",
            "messages": [
                _message(1, False, "周二晚上你去健身房了吗？"),
                _message(2, True, "没去健身房。我去河边散步了。"),
            ],
            "answerable": True,
            "expected_answer": "没有去健身房，而是去河边散步了。",
            "requirements": ["明确回答没有去健身房", "说明实际去河边散步"],
            "forbidden_claims": ["说去了健身房", "把明确的没有说成无法确定"],
        },
        {
            "scenario_id": "S02", "domain": "daily_life", "skills": ["superseded_plan"],
            "collision_subject": "other",
            "question": "这段模拟记录中，我最终确认的手作课是什么时间？",
            "messages": [
                _message(1, True, "周六的手作课先约上午十点。"),
                _message(2, False, "教室上午被占用了，请改到下午两点。"),
                _message(3, True, "确认改成周六下午两点，上午十点的安排取消。"),
            ],
            "answerable": True,
            "expected_answer": "最终确认的是周六下午两点。",
            "requirements": ["给出周六下午两点作为最终安排"],
            "forbidden_claims": ["把上午十点仍列为有效安排", "回答尚未确认"],
        },
        {
            "scenario_id": "S03", "domain": "relationships", "skills": ["subject_attribution", "explicit_negation"],
            "collision_subject": "self",
            "question": "这段模拟记录中，露营地是我订好的，还是聊天对象订好的？我自己的预订是什么状态？",
            "messages": [
                _message(1, False, "我已经订好露营地了。"),
                _message(2, True, "我还没订露营地，等确认假期再订。"),
            ],
            "answerable": True,
            "expected_answer": "聊天对象已经订好露营地；我自己还没订，准备等确认假期再订。",
            "requirements": ["已预订归属于聊天对象", "本人还未预订"],
            "forbidden_claims": ["把聊天对象的预订归给本人", "仅输出作者或归属标签，没有说明预订状态"],
        },
        {
            "scenario_id": "S04", "domain": "work", "skills": ["multipart_question", "subject_attribution"],
            "collision_subject": "other",
            "question": "这段模拟记录中，演示什么时候、在哪里？我需要带什么？",
            "messages": [
                _message(1, True, "演示排在周四上午九点，地点是四楼小会议室；我负责带转接头。"),
                _message(2, False, "收到，我带样机。"),
            ],
            "answerable": True,
            "expected_answer": "演示在周四上午九点、四楼小会议室；我需要带转接头。",
            "requirements": ["周四上午九点", "四楼小会议室", "本人携带转接头"],
            "forbidden_claims": ["把样机说成本人需要携带的物品", "遗漏任一所问子项"],
        },
        {
            "scenario_id": "S05", "domain": "learning", "skills": ["unknown_detail"],
            "collision_subject": "self",
            "question": "这段模拟记录中，陶艺班的学费是多少？",
            "messages": [
                _message(1, False, "下个月的陶艺班已经开放报名了。"),
                _message(2, True, "我准备去了解一下课程，费用和开课日期还没问清楚。"),
            ],
            "answerable": False,
            "expected_answer": "无法确定，记录里还没有学费信息。",
            "requirements": ["明确指出学费未知或记录未提供"],
            "forbidden_claims": ["编造学费数值或范围", "因没有价格就断言免费"],
        },
        {
            "scenario_id": "S06", "domain": "relationships", "skills": ["quotation_attribution", "explicit_negation"],
            "collision_subject": "other",
            "question": "这段模拟记录中，我自己是否已经买好去北城的车票？",
            "messages": [
                _message(1, True, "下面引用的是室友发来的消息。我自己周末还留在本市，没有买去北城的车票。",
                         "我周六去北城，车票已经买好了。"),
                _message(2, False, "知道了，那是室友的行程，不是你的。"),
            ],
            "answerable": True,
            "expected_answer": "没有；已经买好车票的是引用消息里的室友。",
            "requirements": ["明确本人没有买好该车票", "不将引用中的已购票状态归给本人"],
            "forbidden_claims": ["说本人已买好车票", "因为存在引用而忽略本人明确的否定"],
        },
        {
            "scenario_id": "S07", "domain": "learning", "skills": ["superseded_plan", "multipart_question", "exact_detail"],
            "collision_subject": "self",
            "question": "这段模拟记录中，从周三起我每天练习多久，记录放在哪个文件？",
            "messages": [
                _message(1, True, '本周先每天做四十分钟练习，记录放在 "practice-old.txt"。'),
                _message(2, False, "这周任务多，练习时间可以再少些。"),
                _message(3, True, '按新安排执行：从周三起每天二十分钟，记录改放 "practice-new.txt"。'),
            ],
            "answerable": True,
            "expected_answer": "从周三起每天练习二十分钟，记录放在 practice-new.txt。",
            "requirements": ["每天二十分钟", "文件为 practice-new.txt"],
            "forbidden_claims": ["仍采用四十分钟", "仍采用 practice-old.txt", "遗漏时长或文件名"],
        },
        {
            "scenario_id": "S08", "domain": "communication", "skills": ["temporary_state", "stable_preference", "multipart_question"],
            "collision_subject": "other",
            "question": "这段模拟记录中，今晚我希望怎样交流？平时讨论作品的偏好有没有改变？",
            "messages": [
                _message(1, True, "讨论作品时，我通常希望你先指出问题，再给修改建议。"),
                _message(2, False, "今晚也按这个方式聊吗？"),
                _message(3, True, "今晚有点累，先只听我说就好；这只限今晚，平时讨论作品仍按原来的方式。"),
            ],
            "answerable": True,
            "expected_answer": "今晚希望先只听我说；平时讨论作品仍先指出问题、再给修改建议，长期偏好没有改变。",
            "requirements": ["今晚先倾听", "平时仍先指出问题再给修改建议", "不将今晚要求永久化"],
            "forbidden_claims": ["说以后都不应指出问题", "忽略今晚的临时请求", "把临时疲惫写成永久性格"],
        },
    ]


def _name(message, collision_subject, level):
    subject = "self" if message["is_self"] else "other"
    if subject == collision_subject and level == "assistant":
        return "assistant"
    return "青禾" if message["is_self"] else "远舟"


def _string(value, serialization):
    return json.dumps(value, ensure_ascii=False) if serialization == "json" else value


def validate_factors(factors):
    if set(factors) != set(FACTORS):
        raise ValueError("factor names differ from the frozen factorial design")
    for name, levels in FACTORS.items():
        value = factors[name]
        if not any(type(value) is type(level) and value == level for level in levels):
            raise ValueError("invalid factor level: " + name)


def render_message(message, factors, collision_subject, *, include_display_name=True):
    """Render a source record without access to the question, answer or rubric."""
    validate_factors(factors)
    if type(message["is_self"]) is not bool:
        raise ValueError("the authoritative source role must be boolean")
    if collision_subject not in ("self", "other"):
        raise ValueError("invalid collision subject")
    role = "本人" if message["is_self"] else "他人"
    author = ["[作者]", "归属：" + role]
    if factors["extra_is_self_field"]:
        author.append("is_self：" + ("true" if message["is_self"] else "false"))
    if include_display_name:
        author.append("显示名：" + _string(_name(message, collision_subject, factors["display_name_collision"]),
                                        factors["string_serialization"]))
    author.append("[/作者]")
    body = ["正文：" + _string(message["body"], factors["string_serialization"])]
    if message["quoted"]:
        body.append("引用（不自动归属当前发送者）：" + _string(message["quoted"], factors["string_serialization"]))
    middle = author + body if factors["author_block_position"] == "before_body" else body + author
    return "\n".join([f"[记录{message['number']} {message['timestamp']}]", *middle,
                      f"[/记录{message['number']}]"])


def _identity(scenario_id, factors, include_display_name):
    return "syn-" + sha256(canonical_bytes([VERSION, scenario_id, factors, include_display_name]))[:20]


def _projection(scenario):
    return {key: deepcopy(scenario[key]) for key in ("scenario_id", "domain", "skills", "collision_subject", "question", "messages")}


def _model_case(scenario, factors, include_display_name):
    # No expected_answer, requirements, answerable or forbidden_claims enter rendering.
    return {
        "id": _identity(scenario["scenario_id"], factors, include_display_name),
        "question": scenario["question"],
        "correct": [render_message(message, factors, scenario["collision_subject"],
                                   include_display_name=include_display_name)
                    for message in scenario["messages"]],
    }


def build_experiment(scenarios=None):
    scenarios = synthetic_scenarios() if scenarios is None else deepcopy(scenarios)
    if len(scenarios) != 8 or len({s["scenario_id"] for s in scenarios}) != 8:
        raise ValueError("the frozen design requires eight unique synthetic scenarios")
    if sum(s["collision_subject"] == "self" for s in scenarios) != 4:
        raise ValueError("the display-name collision must be balanced across source roles")
    conditions = [dict(zip(FACTORS, levels)) for levels in itertools.product(*FACTORS.values())]
    cases, mapping, keys, contrasts = [], [], [], []
    for scenario in scenarios:
        if {m["is_self"] for m in scenario["messages"]} != {False, True}:
            raise ValueError("every scenario requires both source roles")
        canonical_source = _projection(scenario)
        for factors, include_name in [(f, True) for f in conditions] + [(DEFAULT_FACTORS, False)]:
            case = _model_case(canonical_source, factors, include_name)
            cases.append(case)
            mapping.append({
                "id": case["id"], "scenario_id": scenario["scenario_id"],
                "factors": dict(factors), "include_display_name": include_name,
                "collision_subject": scenario["collision_subject"],
                "source_messages_sha256": sha256(canonical_bytes(scenario["messages"])),
                "question_sha256": sha256(case["question"].encode()),
                "evidence_sha256": sha256(canonical_bytes(case["correct"])),
            })
            keys.append({"id": case["id"], "scenario_id": scenario["scenario_id"],
                         **{name: deepcopy(scenario[name]) for name in
                            ("answerable", "expected_answer", "requirements", "forbidden_claims")},
                         "grading_rule": "All required facts and subject assignments must hold; no forbidden claim. Equivalent natural Chinese is accepted. Do not require verbatim wording."})
        for factor, levels in FACTORS.items():
            others = [name for name in FACTORS if name != factor]
            for fixed in itertools.product(*(FACTORS[name] for name in others)):
                left = dict(zip(others, fixed)); left[factor] = levels[0]
                right = dict(left); right[factor] = levels[1]
                contrasts.append({
                    "scenario_id": scenario["scenario_id"], "factor": factor,
                    "left_id": _identity(scenario["scenario_id"], left, True),
                    "right_id": _identity(scenario["scenario_id"], right, True),
                    "left_level": levels[0], "right_level": levels[1],
                    "fixed_factors": {name: left[name] for name in others},
                    "include_display_name": True,
                })
        contrasts.append({
            "scenario_id": scenario["scenario_id"], "factor": "display_name_presence",
            "left_id": _identity(scenario["scenario_id"], DEFAULT_FACTORS, False),
            "right_id": _identity(scenario["scenario_id"], DEFAULT_FACTORS, True),
            "left_level": False, "right_level": True, "fixed_factors": dict(DEFAULT_FACTORS),
        })
    # Opaque IDs and a deterministic mixed order keep readable condition names out
    # of input records. Presentation itself can still reveal the condition to a judge.
    result = {
        "cases": sorted(cases, key=lambda row: row["id"]),
        "condition_map": sorted(mapping, key=lambda row: row["id"]),
        "answer_key": sorted(keys, key=lambda row: row["id"]),
        "contrasts": sorted(contrasts, key=lambda row: (row["factor"], row["scenario_id"], row["left_id"])),
        "source_catalog": [_projection(s) for s in scenarios],
    }
    validate_design(result)
    return result


def validate_design(result):
    cases = {row["id"]: row for row in result["cases"]}
    maps = {row["id"]: row for row in result["condition_map"]}
    keys = {row["id"]: row for row in result["answer_key"]}
    sources = {row["scenario_id"]: row for row in result["source_catalog"]}
    if (any(len(result[name]) != 136 for name in ("cases", "condition_map", "answer_key"))
            or len(cases) != 136 or set(cases) != set(maps) or set(cases) != set(keys)
            or len(sources) != 8 or len(result["contrasts"]) != 264):
        raise ValueError("case, key and condition IDs must align one to one")
    for row in result["cases"]:
        if set(row) != {"id", "question", "correct"}:
            raise ValueError("reader case contains a non-input field")
        mapping = maps[row["id"]]
        source = sources[mapping["scenario_id"]]
        if row != _model_case(source, mapping["factors"], mapping["include_display_name"]):
            raise ValueError("rendered input differs from its fixed source and declared factors")
        if (mapping["source_messages_sha256"] != sha256(canonical_bytes(source["messages"]))
                or mapping["question_sha256"] != sha256(row["question"].encode())
                or mapping["evidence_sha256"] != sha256(canonical_bytes(row["correct"]))):
            raise ValueError("input fingerprints differ from actual content")
    expected_contrast_keys = set()
    for scenario_id in sources:
        for factor in FACTORS:
            others = [name for name in FACTORS if name != factor]
            for fixed in itertools.product(*(FACTORS[name] for name in others)):
                expected_contrast_keys.add((scenario_id, factor, canonical_bytes(dict(zip(others, fixed)))))
        expected_contrast_keys.add((scenario_id, "display_name_presence", canonical_bytes(DEFAULT_FACTORS)))
    seen_contrast_keys = set()
    for pair in result["contrasts"]:
        if pair.get("left_id") not in maps or pair.get("right_id") not in maps:
            raise ValueError("contrast refers to an absent input")
        a, b = maps[pair["left_id"]], maps[pair["right_id"]]
        if a["scenario_id"] != b["scenario_id"] or a["source_messages_sha256"] != b["source_messages_sha256"]:
            raise ValueError("contrast changes source semantics")
        if cases[a["id"]]["question"] != cases[b["id"]]["question"]:
            raise ValueError("contrast changes the question")
        if {k: v for k, v in keys[a["id"]].items() if k != "id"} != {k: v for k, v in keys[b["id"]].items() if k != "id"}:
            raise ValueError("contrast changes the answer or rubric")
        delta = [name for name in FACTORS if a["factors"][name] != b["factors"][name]]
        if a["include_display_name"] != b["include_display_name"]:
            delta.append("display_name_presence")
        if delta != [pair["factor"]]:
            raise ValueError("contrast must change exactly its declared factor")
        factor = pair["factor"]
        if factor in FACTORS:
            expected = {"scenario_id": a["scenario_id"], "factor": factor,
                        "left_id": a["id"], "right_id": b["id"],
                        "left_level": FACTORS[factor][0], "right_level": FACTORS[factor][1],
                        "fixed_factors": {name: a["factors"][name] for name in FACTORS if name != factor},
                        "include_display_name": True}
            if (a["factors"][factor] != FACTORS[factor][0] or b["factors"][factor] != FACTORS[factor][1]
                    or not a["include_display_name"] or not b["include_display_name"]):
                raise ValueError("contrast direction differs from its declared factor levels")
        elif factor == "display_name_presence":
            expected = {"scenario_id": a["scenario_id"], "factor": factor,
                        "left_id": a["id"], "right_id": b["id"],
                        "left_level": False, "right_level": True,
                        "fixed_factors": dict(DEFAULT_FACTORS)}
            if a["include_display_name"] is not False or b["include_display_name"] is not True:
                raise ValueError("display-name presence contrast has the wrong direction")
        else:
            raise ValueError("unknown contrast factor")
        if canonical_bytes(pair) != canonical_bytes(expected):
            raise ValueError("contrast metadata differs from its actual inputs")
        contrast_key = (pair["scenario_id"], factor, canonical_bytes(pair["fixed_factors"]))
        if contrast_key in seen_contrast_keys:
            raise ValueError("duplicate declared contrast")
        seen_contrast_keys.add(contrast_key)
        if cases[a["id"]]["correct"] == cases[b["id"]]["correct"]:
            raise ValueError("declared contrast does not alter the rendered input")
    if seen_contrast_keys != expected_contrast_keys:
        raise ValueError("declared contrasts do not exhaust the frozen factorial design")


def _jsonl(rows):
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def write_experiment(output, *, reader_file):
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new output directory")
    reader_hash = sha256(Path(reader_file).read_bytes())
    result = build_experiment()
    files = {name + ".jsonl": _jsonl(result[name]) for name in
             ("cases", "condition_map", "answer_key", "contrasts")}
    files["source_catalog.json"] = canonical_bytes({"synthetic": True, "scenarios": result["source_catalog"]}) + b"\n"
    evidence_sizes = [len("\n".join(row["correct"])) for row in result["cases"]]
    manifest = {
        "version": VERSION, "status": "prepared_not_executed", "synthetic": True,
        "model_generation_run": False, "model_answers": 0, "model_accuracy": None,
        "actual_input_tokens": None, "training_run": False, "personalization_validated": False,
        "scenarios": 8, "factorial_conditions_per_scenario": 16,
        "no_display_name_controls": 8, "cases": len(result["cases"]),
        "paired_contrasts": len(result["contrasts"]),
        "factor_levels": FACTORS, "no_display_name_fixed_factors": DEFAULT_FACTORS,
        "quote_scope": ["display_name", "body", "quoted"],
        "unchanged_fields": ["question", "message order", "source timestamp", "source role", "body", "quoted", "reference answer", "grading rubric"],
        "collision_subject_balance": {"self": 4, "other": 4},
        "evidence_unicode_characters": {"min": min(evidence_sizes), "max": max(evidence_sizes)},
        "generation_protocol": {"mode": "research", "prompt_version": "original", "conditions": ["correct"],
                                "do_sample": False, "enable_thinking": False,
                                "max_input_tokens": 4096, "max_new_tokens": 192,
                                "adapter": None, "model": "same pinned existing local baseline; verify at execution"},
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "reader_sha256": reader_hash,
        "files": {name: {"sha256": sha256(value), "bytes": len(value)} for name, value in files.items()},
        "analysis_plan": [
            "Use every declared within-scenario contrast; do not select only successful pairs.",
            "For each factor, report corrected, regressed, unchanged-correct and unchanged-wrong counts, plus eight scenario-level summaries.",
            "Report name-collision effects separately when the collision belongs to self and to other.",
            "Report substantive answer correctness, unsupported claims, subject-attribution errors and metadata-only output separately.",
            "A correct refusal on the unknown-detail scenario is distinct from refusing answerable questions.",
            "Variants from one scenario are correlated; 136 inputs are not 136 independent tasks.",
        ],
        "limitations": [
            "Researcher-authored synthetic development diagnostic, not a personal-memory or independent generalization acceptance set.",
            "The fixed record/author boundaries define a controlled synthetic format, not a byte-for-byte reproduction of an earlier production format. Effects apply within this representation.",
            "String serialization applies jointly to display name, body and quoted fields; it does not isolate a single quote character or model-internal mechanism.",
            "Display-name substitution changes string identity and tokenization together. No equal-token or equal-character-budget claim is made.",
            "Extra is_self only repeats the authoritative role; it never changes the actor or the expected answer.",
            "Self/other collision groups use different scenarios. Their separate summaries are descriptive; collision subject is not a fifth crossed factor and does not establish which actor is more vulnerable.",
            "Source text is complete in every variant. No retrieval, selection, clipping or memory update occurs.",
            "Opaque IDs do not hide visibly different formats from an evaluator.",
            "Outcomes are not available until the pinned local model has actually run; do not tune this frozen generator against real-question outcomes.",
        ],
    }
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    (output / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reader-file", type=Path, required=True,
                        help="Exact existing reader source file to hash; never executed")
    args = parser.parse_args()
    manifest = write_experiment(args.out, reader_file=args.reader_file)
    print(json.dumps({"cases": manifest["cases"], "paired_contrasts": manifest["paired_contrasts"],
                      "status": manifest["status"], "output": str(args.out)}))


if __name__ == "__main__":
    main()

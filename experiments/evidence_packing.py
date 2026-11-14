"""Pack retrieved original messages into a bounded, auditable evidence payload.

No retrieval, model, network, gold labels, paraphrasing or memory writes occur.
The budget measures the final joined evidence text in Unicode code points, not
model tokens. Input messages must already obey the caller's visibility rules.
Source offsets refer to the original body/quoted strings before JSON escaping.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import re
import unicodedata


VERSION = "evidence-packing-v08-1"
SHORT_TEXT_CHARS = 240
MAX_FIELD_SPANS = 3
FIELDS = ("body", "quoted")
NOISE = {"什么", "多少", "哪个", "哪里", "一下", "请问", "这个", "那个", "我们",
         "你们", "之前", "之后", "现在", "记录", "时候", "的是", "怎么", "我的"}
BOUNDARY = re.compile(r'[。！？!?；;\n]+[”’」』）)\"]*|\.(?=\s|$)')


def _json(text):
    # Metadata and utterances stay on labelled lines; embedded newlines cannot
    # create fake author headings. The unescaped originals remain in the audit.
    return json.dumps(text, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _terms(text):
    values = set()
    for run in re.findall(r"[\u3400-\u9fff]+|[a-z0-9_]+", unicodedata.normalize("NFKC", text).casefold()):
        if run.isascii():
            values.add(run)
        elif len(run) > 1:
            values.update(run[i:i + 2] for i in range(len(run) - 1))
        elif run not in {"我", "你", "他", "她", "的", "了", "吗", "呢"}:
            values.add(run)
    return values - NOISE


def _sentences(text):
    result, start = [], 0
    for match in BOUNDARY.finditer(text):
        result.append((start, match.end()))
        start = match.end()
    if start < len(text):
        result.append((start, len(text)))
    return result


def _merge(spans):
    result = []
    for start, end in sorted(set(spans)):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _omitted(length, spans):
    result, cursor = [], 0
    for start, end in _merge(spans):
        if cursor < start:
            result.append((cursor, start))
        cursor = end
    if cursor < length:
        result.append((cursor, length))
    return result


def _validate_record(value, archive_sha256):
    if not isinstance(value, Mapping):
        raise ValueError("each message must be an object")
    if type(value.get("line")) is not int or value["line"] < 1:
        raise ValueError("line must be a positive source-line integer")
    if type(value.get("is_self")) not in (bool, int) or value["is_self"] not in (0, 1):
        raise ValueError("is_self must be boolean or 0/1")
    timestamp = value.get("ts")
    if not isinstance(timestamp, str):
        raise ValueError("ts must be an offset-aware ISO timestamp")
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("ts requires a UTC offset")
    for key in ("source_sha256", "source_archive_sha256", "zip_sha256"):
        if value.get(key) is not None and value[key] != archive_sha256:
            raise ValueError("message and requested archive hashes differ")
    result = {"line": value["line"], "ts": timestamp, "is_self": bool(value["is_self"])}
    for key in FIELDS:
        if key not in value:
            raise ValueError(key + " is required, including an explicit empty string")
        content = value[key]
        if not isinstance(content, str):
            raise ValueError(key + " must be the original string")
        result[key] = content
    for key in ("chat_id", "chat_name", "sender", "sender_id"):
        content = value.get(key) or ""
        if not isinstance(content, str):
            raise ValueError(key + " must be text when provided")
        result[key] = content
    return result


def _deduplicate(messages, archive_sha256):
    result, positions = [], {}
    for raw in messages:
        row = _validate_record(raw, archive_sha256)
        if row["line"] not in positions:
            positions[row["line"]] = len(result)
            result.append(row)
            continue
        previous = result[positions[row["line"]]]
        for key in ("ts", "is_self", *FIELDS):
            if row[key] != previous[key]:
                raise ValueError("conflicting content for the same source line")
        for key in ("chat_id", "chat_name", "sender", "sender_id"):
            if previous[key] and row[key] and previous[key] != row[key]:
                raise ValueError("conflicting metadata for the same source line")
            previous[key] = previous[key] or row[key]
    return result


def _render_record(row, selected):
    name = row["sender"] or "未提供姓名"
    role = "本人" if row["is_self"] else "他人"
    # The model gets a short source handle; the separate audit maps it to the
    # full archive identity, author ID and exact offsets for expansion.
    header = f"[消息#{row['line']}] {_json(row['ts'])} {role} {_json(name)}"
    parts = [header]
    for field in FIELDS:
        text = row[field]
        label = "正文" if field == "body" else "引用（原作者未核实）"
        if not text:
            if field == "body":
                parts.append("正文为空。")
            continue
        spans = _merge(selected[field])
        cursor, pieces = 0, []
        for start, end in spans:
            if start > cursor:
                pieces.append("〔省略〕")
            pieces.append(_json(text[start:end]))
            cursor = end
        if cursor < len(text):
            pieces.append("〔省略〕")
        parts.append(label + "：" + " ".join(pieces))
    return "\n".join(parts)


def _full(row):
    return {field: [(0, len(row[field]))] if row[field] else [] for field in FIELDS}


def _choose_record(row, query_terms, limit):
    """Select exact source spans; the return value is never a factual summary."""
    complete = _full(row)
    if len(_render_record(row, complete)) <= limit:
        return complete, ["complete_message"]
    selected = {field: [] for field in FIELDS}
    if len(_render_record(row, selected)) > limit:
        return None
    methods, candidates = [], []
    for field in FIELDS:
        text, sentences = row[field], _sentences(row[field])
        for index, (start, end) in enumerate(sentences):
            matches = query_terms & _terms(text[start:end])
            if not matches:
                continue
            # Adjacent original sentences preserve local context when they fit.
            lo = sentences[max(0, index - 1)][0]
            hi = sentences[min(len(sentences) - 1, index + 1)][1]
            weight = sum(max(1, min(len(token), 12)) for token in matches)
            candidates.append((weight / (1 + (hi - lo) / 240), True, field, lo, hi, matches))
            candidates.append((weight / (1 + (end - start) / 240), False, field, start, end, matches))
    covered = set()
    # Choose a complete context block before its single-sentence fallback when
    # both fit. Recompute marginal query coverage to support multipart questions.
    while candidates:
        candidates.sort(key=lambda c: (-sum(max(1, min(len(x), 12)) for x in c[5] - covered),
                                       not c[1], -c[0], FIELDS.index(c[2]), c[3]))
        _, context, field, start, end, matches = candidates.pop(0)
        proposal = {key: list(value) for key, value in selected.items()}
        proposal[field] = _merge([*proposal[field], (start, end)])
        if proposal == selected or len(proposal[field]) > MAX_FIELD_SPANS:
            continue
        if len(_render_record(row, proposal)) <= limit:
            selected = proposal
            covered.update(matches)
            methods.append("query_sentence_with_neighbors" if context else "query_sentence")
    # Use remaining capacity conservatively. Without a lexical match this is an
    # explicit head/tail excerpt, never a claim that the missing middle is absent.
    for field in FIELDS:
        text = row[field]
        if not text:
            continue
        proposal = {key: list(value) for key, value in selected.items()}
        proposal[field] = [(0, len(text))]
        if len(_render_record(row, proposal)) <= limit:
            selected = proposal
            methods.append("complete_field")
            continue
        low, high, best = 1, len(text) - 1, None
        while low <= high:
            count = (low + high) // 2
            head, tail = (count + 1) // 2, count // 2
            proposal = {key: list(value) for key, value in selected.items()}
            proposal[field] = _merge([*proposal[field], (0, head), (len(text) - tail, len(text))])
            fits = (len(proposal[field]) <= MAX_FIELD_SPANS and
                    len(_render_record(row, proposal)) <= limit)
            if fits:
                best = proposal
                low = count + 1
            else:
                high = count - 1
        if best is not None and best != selected:
            selected = best
            methods.append("explicit_head_tail")
    if any(row[field] for field in FIELDS) and not any(selected.values()):
        return None
    return selected, list(dict.fromkeys(methods))


def _audit_record(row, selected, archive_sha256, *, included, reason, methods):
    result = {key: row[key] for key in ("line", "ts", "is_self", "chat_id", "chat_name", "sender", "sender_id")}
    result.update(source_id=f"{archive_sha256}:{row['line']}", display_source_id=f"消息#{row['line']}",
                  archive_sha256=archive_sha256,
                  included=included, omission_reason=reason, selection_methods=methods,
                  quote_attribution="quoted content; original author is not inferred")
    result["fields"] = {}
    for field in FIELDS:
        text, spans = row[field], _merge(selected[field])
        result["fields"][field] = {
            "source_length": len(text),
            "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "spans": [{"start": lo, "end": hi, "text": text[lo:hi]} for lo, hi in spans],
            "omitted_ranges": [{"start": lo, "end": hi} for lo, hi in _omitted(len(text), spans)],
            "complete": not _omitted(len(text), spans),
            "expand_ref": {"archive_sha256": archive_sha256, "line": row["line"],
                           "field": field, "start": 0, "end": len(text)},
        }
    return result


def pack_evidence(question, messages, *, archive_sha256, max_chars, max_messages=10):
    """Return evidence:list[str] for evidence_reader and a separate source audit.

    Kept records retain retrieval order. Whole short messages receive allocation
    first; longer messages share the remaining budget, then use leftover space.
    All labels, omission markers and joining newlines count toward max_chars.
    This function does not verify the archive bytes or perform time filtering:
    the caller supplies source-verified, already-visible retrieved records.
    """
    if not isinstance(question, str):
        raise ValueError("question must be text")
    if not isinstance(archive_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", archive_sha256):
        raise ValueError("archive_sha256 must be a lowercase SHA256")
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a nonnegative integer")
    if type(max_messages) is not int or not 1 <= max_messages <= 100:
        raise ValueError("max_messages must be 1..100")
    rows = _deduplicate(messages, archive_sha256)
    eligible = rows[:max_messages]
    intro = "原文片段（〔省略〕处可回查）："
    selections = {}
    remaining = max(0, max_chars - len(intro))
    for index, row in enumerate(eligible):
        if sum(len(row[field]) for field in FIELDS) > SHORT_TEXT_CHARS:
            continue
        full = _full(row)
        size = len(_render_record(row, full)) + 1
        if size <= remaining:
            selections[index] = (full, ["complete_short_message"])
            remaining -= size
    pending = [index for index in range(len(eligible)) if index not in selections]
    query_terms = _terms(question)
    for position, index in enumerate(pending):
        allocation = max(0, remaining // (len(pending) - position) - 1)
        picked = _choose_record(eligible[index], query_terms, allocation)
        if picked is not None:
            selections[index] = picked
            remaining -= len(_render_record(eligible[index], picked[0])) + 1
    # Reuse unused space without changing retrieval order or reranking messages.
    for index, row in enumerate(eligible):
        previous = selections.get(index)
        previous_size = len(_render_record(row, previous[0])) + 1 if previous else 0
        picked = _choose_record(row, query_terms, max(0, remaining + previous_size - 1))
        if picked is not None:
            new_size = len(_render_record(row, picked[0])) + 1
            if new_size <= remaining + previous_size:
                selections[index] = picked
                remaining += previous_size - new_size
    evidence = ([intro] + [_render_record(eligible[index], selections[index][0]) for index in sorted(selections)]
                if selections else [])
    rendered = "\n".join(evidence)
    if len(rendered) > max_chars:
        raise AssertionError("evidence budget exceeded")
    audit = []
    for index, row in enumerate(rows):
        selected, methods = selections.get(index, ({field: [] for field in FIELDS}, []))
        included = index in selections
        reason = None if included else ("max_messages" if index >= max_messages else "character_budget")
        audit.append(_audit_record(row, selected, archive_sha256, included=included, reason=reason, methods=methods))
    return {"version": VERSION, "evidence": evidence, "rendered_text": rendered, "records": audit,
            "empty_reason": ("no_retrieved_messages" if not rows else
                             "character_budget" if not selections else None),
            "budget": {"unit": "unicode_code_points", "maximum": max_chars, "used": len(rendered),
                       "includes": "source labels, quoted string encoding, omission markers and joining newlines"},
            "offset_unit": "unicode_code_points_in_original_field",
            "input_messages": len(rows), "supplied_messages": len(selections),
            "retrieval_order_preserved": True,
            "limitations": ["A source excerpt is not a complete fact or a correctness judgment.",
                            "Lexical sentence selection can miss paraphrases and cross-message dependencies.",
                            "An omitted span may contain a correction, denial or required detail.",
                            "The character ceiling is not a tokenizer or context-window guarantee."]}


def pack_windows(question, windows, **kwargs):
    """Adapt contextual_lookup windows without changing their retrieval order."""
    messages = []
    for window in windows:
        if not isinstance(window, Mapping):
            raise ValueError("each window must be an object")
        for raw in window.get("context", [window]):
            row = dict(raw)
            if window.get("chat_id"):
                if row.get("chat_id") and row["chat_id"] != window["chat_id"]:
                    raise ValueError("context message belongs to a different chat")
                row.setdefault("chat_id", window["chat_id"])
            # Do not inherit the anchor's sender: neighbors can have other authors.
            messages.append(row)
    return pack_evidence(question, messages, **kwargs)

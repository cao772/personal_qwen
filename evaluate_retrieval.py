"""Evaluate a trusted archive_search module on privately stored source-line labels.

This measures evidence retrieval, not model answer accuracy. Queries and archives
remain local; output files are private by default. Use a fresh output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time
from contextlib import closing
from datetime import datetime


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("as_of and source timestamps require UTC offsets")
    return result.timestamp()


def line_set(values):
    if not isinstance(values, list) or any(type(v) is not int or v < 1 for v in values):
        raise ValueError("source lines must be a list of positive integers")
    if len(set(values)) != len(values):
        raise ValueError("duplicate source-line label")
    return set(values)


def load_queries(path):
    queries, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row.get("id"), str) or row["id"] in seen:
                raise ValueError("query IDs must be unique strings")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("empty query")
            if type(row.get("answerable")) is not bool:
                raise ValueError("answerable must be explicitly labelled")
            timestamp(row["as_of"])
            gold = line_set(row["gold_source_lines"])
            excluded = line_set(row.get("excluded_lines", []))
            if row["answerable"] and not gold:
                raise ValueError("answerable query requires source labels")
            if gold & excluded:
                raise ValueError("gold and excluded sources overlap")
            seen.add(row["id"])
            queries.append(row)
    if not queries:
        raise ValueError("empty query file")
    return queries


def canonical_source(db, line, cache):
    if type(line) is not int or line < 1:
        raise ValueError("retriever source line must be a positive integer")
    if line not in cache:
        row = db.execute("SELECT ts,body,quoted FROM message WHERE line=?", (line,)).fetchone()
        if row is None:
            raise ValueError("source-line label does not exist in the indexed archive")
        cache[line] = row
    return cache[line]


def evaluate(module, db, queries, *, k=2, max_messages=10, body_chars=120, quote_chars=60):
    """Only question, as_of and excluded_lines enter the retriever."""
    rows, source_cache = [], {}
    for query in queries:
        cutoff = timestamp(query["as_of"])
        excluded = set(query.get("excluded_lines", []))
        gold = set(query["gold_source_lines"])
        for line in gold:
            source = canonical_source(db, line, source_cache)
            if line in excluded or timestamp(source[0]) > cutoff:
                raise ValueError("gold source is excluded or occurs after as_of")
        started = time.perf_counter()
        hits = module.contextual_lookup(
            db, query["question"], k=k, as_of=query["as_of"], excluded_lines=excluded
        )
        elapsed = time.perf_counter() - started
        if len(hits) > k:
            raise ValueError("retriever exceeded the requested window budget")
        surfaced, seen = [], set()
        for hit in hits:
            for item in hit.get("context", [hit]):
                line = item["line"]
                original = canonical_source(db, line, source_cache)
                if (timestamp(item["ts"]) != timestamp(original[0])
                        or item.get("body", "") != original[1]
                        or item.get("quoted", "") != original[2]):
                    raise ValueError("retrieved source content does not match its indexed line")
                if line in excluded or timestamp(original[0]) > cutoff:
                    raise ValueError("retriever exposed an excluded or future source")
                if line in seen or len(surfaced) >= max_messages:
                    continue
                seen.add(line)
                body = " ".join(item.get("body", "").split())[:body_chars]
                quote = " ".join(item.get("quoted", "").split())[:quote_chars]
                surfaced.append({"line": line, "visible_characters": len(body) + len(quote)})
        matched = gold & seen
        rows.append({
            "id": query["id"],
            "question_sha256": hashlib.sha256(query["question"].encode()).hexdigest(),
            "answerable": query["answerable"],
            "gold_source_lines": sorted(gold),
            "retrieved_source_lines": [r["line"] for r in surfaced],
            "matched_source_lines": sorted(matched),
            "all_labelled_sources": bool(gold <= seen) if query["answerable"] else None,
            "any_labelled_source": bool(matched) if query["answerable"] else None,
            "relevant_context_for_unknown": bool(matched) if not query["answerable"] and gold else None,
            "visible_characters": sum(r["visible_characters"] for r in surfaced),
            "windows": len(hits),
            "seconds": elapsed,
        })
    positive = [r for r in rows if r["answerable"]]
    elapsed = sorted(r["seconds"] for r in rows)
    summary = {
        "metric": "labelled source-line coverage, not answer accuracy",
        "queries": len(rows),
        "answerable_queries": len(positive),
        "unknown_queries": len(rows) - len(positive),
        "all_labelled_sources": sum(r["all_labelled_sources"] for r in positive),
        "any_labelled_source": sum(r["any_labelled_source"] for r in positive),
        "matched_source_lines": sum(len(r["matched_source_lines"]) for r in positive),
        "labelled_source_lines": sum(len(r["gold_source_lines"]) for r in positive),
        "unknown_answer_accuracy": None,
        "budget": {"windows": k, "messages": max_messages,
                   "body_chars_per_message": body_chars, "quote_chars_per_message": quote_chars},
        "latency_seconds": {"median": statistics.median(elapsed),
                            "p95": elapsed[max(0, math.ceil(len(elapsed) * .95) - 1)]},
        "latency_scope": "contextual_lookup only; source-label validation and generation excluded",
        "mean_visible_characters": sum(r["visible_characters"] for r in rows) / len(rows),
        "limitations": [
            "Source-line coverage does not prove that the answer survived text clipping.",
            "Unknown queries require a separate model-response evaluation.",
            "Researcher-authored queries are not representative user acceptance testing.",
            "Label selection may be stricter than the minimum evidence required to answer.",
            "Equal message and character ceilings do not imply equal model-token counts.",
        ],
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True,
                        help="trusted Python file exporting contextual_lookup")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True,
                        help="expected source archive hash from the frozen dataset manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--max-messages", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.windows <= 100 or not 1 <= args.max_messages <= 1000:
        parser.error("invalid evidence budget")
    if len(args.archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.archive_sha256):
        parser.error("--archive-sha256 must be a lowercase SHA256 digest")
    queries = load_queries(args.queries)
    spec = importlib.util.spec_from_file_location("evaluated_archive_search", args.module)
    if spec is None or spec.loader is None:
        parser.error("cannot load retriever")
    module = importlib.util.module_from_spec(spec)
    # Some implementations use dataclasses, which consult the module registry.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        record = db.execute("SELECT value FROM metadata WHERE name='zip_sha256'").fetchone()
        archive_hash = record[0] if record else None
        if archive_hash != args.archive_sha256:
            raise ValueError("indexed archive does not match the frozen dataset source hash")
        for query in queries:
            expected = query.get("source_archive_sha256")
            if expected is not None and expected != archive_hash:
                raise ValueError("query labels and indexed archive have different hashes")
        rows, summary = evaluate(module, db, queries, k=args.windows,
                                 max_messages=args.max_messages)
    summary.update(evaluator_sha256=sha256(__file__), module_sha256=sha256(args.module),
                   queries_sha256=sha256(args.queries),
                   source_archive_sha256=archive_hash)
    os.umask(0o077)
    args.out.mkdir(parents=True, mode=0o700, exist_ok=False)
    (args.out / "retrieval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

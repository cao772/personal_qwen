"""Synthetic checks for evidence-metric integrity; no private conversations."""
import unittest
import sqlite3
from types import SimpleNamespace

from evaluate_retrieval import evaluate, timestamp


def source(line, *, time="2025-03-10T09:00:00+08:00"):
    return {"line": line, "ts": time, "body": "合成的花园活动记录", "quoted": ""}


def query(*, answerable=True, gold=None, excluded=None):
    return {"id": "synthetic", "question": "园艺活动要带什么？",
            "as_of": "2025-03-10T10:00:00+08:00", "answerable": answerable,
            "gold_source_lines": [2] if gold is None else gold,
            "excluded_lines": [] if excluded is None else excluded,
            "reference": "此字段不得进入检索器"}


class EvaluationIntegrity(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("CREATE TABLE message(line INTEGER PRIMARY KEY,ts TEXT,body TEXT,quoted TEXT)")
        for line in [1, 2]:
            row = source(line)
            self.db.execute("INSERT INTO message VALUES(?,?,?,?)", (
                row["line"], row["ts"], row["body"], row["quoted"]))

    def test_only_query_and_visibility_controls_reach_retriever(self):
        calls = []

        def retrieve(db, text, **kwargs):
            calls.append((db, text, kwargs))
            return [{"context": [source(1), source(2)]}]

        rows, summary = evaluate(SimpleNamespace(contextual_lookup=retrieve), self.db, [query()])
        self.assertEqual(calls, [(self.db, "园艺活动要带什么？", {
            "k": 2, "as_of": "2025-03-10T10:00:00+08:00", "excluded_lines": set()})])
        self.assertTrue(rows[0]["all_labelled_sources"])
        self.assertEqual(summary["all_labelled_sources"], 1)

    def test_budget_and_duplicates_cannot_inflate_source_coverage(self):
        retriever = SimpleNamespace(contextual_lookup=lambda *a, **k: [
            {"context": [source(1), source(1), source(2)]}])
        rows, summary = evaluate(retriever, self.db, [query()], max_messages=1)
        self.assertEqual(rows[0]["retrieved_source_lines"], [1])
        self.assertFalse(rows[0]["all_labelled_sources"])
        self.assertEqual(summary["matched_source_lines"], 0)

    def test_unknown_is_not_an_automatically_correct_answer(self):
        retriever = SimpleNamespace(contextual_lookup=lambda *a, **k: [])
        rows, summary = evaluate(retriever, self.db, [query(answerable=False, gold=[])])
        self.assertIsNone(rows[0]["all_labelled_sources"])
        self.assertIsNone(summary["unknown_answer_accuracy"])
        self.assertEqual(summary["answerable_queries"], 0)

    def test_excluded_or_future_sources_fail_even_outside_output_budget(self):
        for bad, excluded in [(source(2), [2]),
                              (source(2, time="2025-03-11T09:00:00+08:00"), [])]:
            with self.subTest(excluded=excluded):
                self.db.execute("UPDATE message SET ts=? WHERE line=2", (bad["ts"],))
                retriever = SimpleNamespace(contextual_lookup=lambda *a, **k: [
                    {"context": [source(1), bad]}])
                with self.assertRaisesRegex(ValueError, "excluded or future"):
                    evaluate(retriever, self.db, [query(gold=[1], excluded=excluded)], max_messages=1)

    def test_fabricated_content_cannot_claim_a_gold_source_id(self):
        modified = source(2)
        modified["body"] = "合成的另一个无关事件"
        retriever = SimpleNamespace(contextual_lookup=lambda *a, **k: [{"context": [modified]}])
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate(retriever, self.db, [query()])

    def test_missing_or_future_gold_source_fails(self):
        retriever = SimpleNamespace(contextual_lookup=lambda *a, **k: [])
        with self.assertRaisesRegex(ValueError, "does not exist"):
            evaluate(retriever, self.db, [query(gold=[99])])
        self.db.execute("UPDATE message SET ts='2025-03-11T09:00:00+08:00' WHERE line=2")
        with self.assertRaisesRegex(ValueError, "gold source"):
            evaluate(retriever, self.db, [query()])

    def test_naive_time_is_rejected(self):
        with self.assertRaises(ValueError):
            timestamp("2025-03-10T10:00:00")


if __name__ == "__main__":
    unittest.main()

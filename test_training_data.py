"""Dependency-free checks for leakage and chronology in the training entry."""
import unittest

from train_memory_prefix import check_split


def row(question, memory, time, identifier=None):
    result = {"question": question, "answer": "answer", "memories": [memory],
              "timestamp": time, "_location": "synthetic-test"}
    if identifier is not None:
        result["id"] = identifier
    return result


class TrainingSplitTests(unittest.TestCase):
    def test_chronology_uses_utc_not_lexical_timestamp(self):
        train = [row("q1", "a", "2001-01-01T08:30:00+08:00")]
        evaluation = [row("q2", "b", "2001-01-01T01:00:00Z")]
        check_split(train, evaluation, "chronological")

    def test_future_training_record_is_rejected(self):
        train = [row("q1", "a", "2001-01-02T00:00:00Z")]
        evaluation = [row("q2", "b", "2001-01-01T23:00:00Z")]
        with self.assertRaisesRegex(ValueError, "chronological"):
            check_split(train, evaluation, "chronological")

    def test_answer_edit_cannot_hide_duplicate_inputs(self):
        train = [row("same question", "same memory", "2001-01-01T00:00:00Z")]
        evaluation = [row("same   question", "same memory", "2001-01-02T00:00:00Z")]
        evaluation[0]["answer"] = "a different answer does not make this held out"
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_split(train, evaluation, "chronological")

    def test_timezone_is_required(self):
        train = [row("q1", "a", "2001-01-01T00:00:00")]
        evaluation = [row("q2", "b", "2001-01-02T00:00:00Z")]
        with self.assertRaisesRegex(ValueError, "timezone"):
            check_split(train, evaluation, "chronological")


if __name__ == "__main__":
    unittest.main()

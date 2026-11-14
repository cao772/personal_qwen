"""Previously failing counterexamples must pass for candidate B."""
from contextlib import closing
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from experiments import archive_search_candidate_b as candidate


class CandidateBFixedLimits(unittest.TestCase):
    def test_date_predicate_attachment(self):
        cutoff = "2031-02-15T12:00:00+08:00"
        for question in ("记录里提到的2月9日活动安排是什么？",
                         "之前说过2月9日的活动吗？", "明天的活动昨天安排了吗？"):
            with self.subTest(question=question):
                self.assertFalse(candidate.parse_query_time(question, cutoff)["filter_applied"])
        for question in ("2月9日我说过什么？", "2月9日，群里聊了什么？",
                         "2月9日到2月10日的聊天", "昨天聊天提到什么？"):
            with self.subTest(question=question):
                self.assertTrue(candidate.parse_query_time(question, cutoff)["filter_applied"])

    def test_activity_date_with_record_words_is_not_a_message_date(self):
        info = candidate.parse_query_time(
            "之前记录里提到的2月9日活动安排是什么？", "2031-02-05T12:00:00+08:00")
        # A plan sent on Feb 1 remains valid evidence for an activity on Feb 9.
        self.assertFalse(info["filter_applied"])

    def test_merged_context_honors_its_declared_anchor_horizon(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            start = datetime.fromisoformat("2031-02-01T10:00:00+08:00")
            rows = [dict(chat_id="synthetic-chat", chat_name="SYNTHETIC",
                         local_id=i + 1, time=(start + timedelta(seconds=offset)).isoformat(),
                         is_self=False, text="合成活动记录", quoted_text="")
                    for i, offset in enumerate([0, 100, 200, 500, 700, 1000])]
            source, index = folder / "source.zip", folder / "archive.sqlite"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("messages.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
            candidate.build(source, index)
            with closing(sqlite3.connect(index)) as db:
                seeds = candidate._dicts(db.execute("SELECT * FROM message WHERE line IN (1,4) ORDER BY line"))
                for seed in seeds:
                    seed["source_sha256"] = "0" * 64
                time_info = candidate.parse_query_time("合成活动", "2031-02-01T20:00:00+08:00")
                # Isolate the window merge contract from the ranking algorithm.
                with patch.object(candidate, "_rank", return_value=(seeds, time_info)):
                    hits = candidate.contextual_lookup(db, "合成活动", k=1,
                                                       as_of="2031-02-01T20:00:00+08:00")
                self.assertTrue(all(abs(raw["epoch"] - hit["epoch"]) <= 600
                                    for hit in hits for raw in hit["context"]))


if __name__ == "__main__":
    unittest.main()

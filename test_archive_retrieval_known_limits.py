"""Synthetic counterexamples for frozen candidate A; expected failures are intentional.

These document why this archived experiment must not become the daily default.
Fix a separate candidate and preserve A's original bytes for reproducibility.
"""
from contextlib import closing
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from experiments import archive_search_candidate_a as candidate


class FrozenCandidateLimitations(unittest.TestCase):
    @unittest.expectedFailure
    def test_activity_date_with_record_words_is_not_a_message_date(self):
        info = candidate.parse_query_time(
            "之前记录里提到的2月9日活动安排是什么？", "2031-02-05T12:00:00+08:00")
        # A plan sent on Feb 1 remains valid evidence for an activity on Feb 9.
        self.assertFalse(info["filter_applied"])

    @unittest.expectedFailure
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

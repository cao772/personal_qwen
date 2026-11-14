"""Candidate B synthetic behavior tests (no personal data)."""
from contextlib import closing
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

from experiments.archive_search_candidate_b import build, lookup, contextual_lookup, parse_query_time


def message(chat, local, at, text, *, quote="", own=False, sender="SYNTHETIC PERSON"):
    return dict(chat_id=chat,chat_name="SYNTHETIC "+chat,chat_type="group",local_id=local,
                time=at,is_self=own,text=text,quoted_text=quote,sender=sender,
                sender_id="synthetic-sender",type="普通文本",server_id="synthetic-server-"+str(local))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def index(self, rows):
        self.counter += 1
        source = self.root/f"source-{self.counter}.zip"
        dest = self.root/f"archive-{self.counter}.sqlite"
        with zipfile.ZipFile(source,"w") as archive:
            archive.writestr("messages.jsonl","".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
        build(source,dest)
        return sqlite3.connect(dest),source,dest

    def test_actual_same_chat_neighbors_across_interleaved_lines(self):
        rows = [message("A",1,"2031-02-03T10:00:00+08:00","问题是想去哪个场地",own=True)]
        rows += [message("B",i,"2031-02-03T10:00:10+08:00","无关内容") for i in range(2,12)]
        rows += [message("A",2,"2031-02-03T10:00:20+08:00","蓝鲸计划选择西苑场地"),
                 message("A",3,"2031-02-03T10:00:30+08:00","确认就这样安排",own=True)]
        db,_,_ = self.index(rows)
        with closing(db):
            hits = contextual_lookup(db,"蓝鲸计划场地",k=2,as_of="2031-02-04T00:00:00+08:00")
            self.assertEqual({r["line"] for r in hits[0]["context"]},{1,12,13})
            self.assertTrue(all(r["chat_id"] == "A" for r in hits[0]["context"]))
            self.assertEqual(hits[0]["context"][0]["speaker_role"],"self")
            self.assertIn("sender_id",hits[0]["context"][0])

    def test_future_and_excluded_lines_cannot_enter_context(self):
        rows = [message("A",1,"2031-02-03T10:00:00+08:00","蓝鲸计划旧方案"),
                message("A",2,"2031-02-03T10:00:10+08:00","蓝鲸计划当前方案"),
                message("A",3,"2031-02-03T10:00:20+08:00","蓝鲸计划未来方案")]
        db,_,_ = self.index(rows)
        with closing(db):
            hits = contextual_lookup(db,"蓝鲸计划",k=2,as_of="2031-02-03T10:00:15+08:00",excluded_lines={1})
            self.assertEqual([r["line"] for h in hits for r in h["context"]],[2])
            self.assertTrue(all(h["line"] == 2 for h in hits))

    def test_future_documents_do_not_change_visible_bm25(self):
        base = [message("A",1,"2031-01-01T00:00:00+08:00","蓝鲸计划场地确认"),
                message("A",2,"2031-01-02T00:00:00+08:00","其他计划场地")]
        db1,_,_ = self.index(base)
        future = [message("B",i,"2032-01-01T00:00:00+08:00","蓝鲸计划场地确认") for i in range(30)]
        db2,_,_ = self.index(base+future)
        with closing(db1),closing(db2):
            a = lookup(db1,"蓝鲸计划场地",as_of="2031-02-01T00:00:00+08:00")
            b = lookup(db2,"蓝鲸计划场地",as_of="2031-02-01T00:00:00+08:00")
            self.assertEqual([(r["line"],r["score"]) for r in a],[(r["line"],r["score"]) for r in b])

    def test_relative_time_uses_as_of_and_shanghai_calendar(self):
        rows = [message("A",1,"2031-02-02T23:59:00+08:00","蓝鲸计划确认"),
                message("A",2,"2031-02-03T00:01:00+08:00","蓝鲸计划确认")]
        db,_,_ = self.index(rows)
        with closing(db):
            hits = lookup(db,"昨天聊天提到的蓝鲸计划",as_of="2031-02-02T17:00:00+00:00")
            self.assertEqual([h["line"] for h in hits],[1])
            self.assertTrue(hits[0]["query_time"]["filter_applied"])

    def test_activity_date_is_not_assumed_message_date(self):
        db,_,_ = self.index([message("A",1,"2031-02-01T12:00:00+08:00","定于2月9日举办蓝鲸活动")])
        with closing(db):
            hits = lookup(db,"2月9日蓝鲸活动安排",as_of="2031-02-05T00:00:00+08:00")
            self.assertEqual(hits[0]["line"],1)
            self.assertFalse(hits[0]["query_time"]["filter_applied"])

    def test_explicit_dates_range_and_no_future_evidence(self):
        rows = [message("A",i,f"2031-02-0{i}T12:00:00+08:00","蓝鲸计划") for i in range(1,5)]
        db,_,_ = self.index(rows)
        with closing(db):
            hits = lookup(db,"蓝鲸计划",as_of="2031-02-03T13:00:00+08:00",date_from="2031-02-02",date_to="2031-02-04")
            self.assertEqual({h["line"] for h in hits},{2,3})
            parsed = parse_query_time("2031年2月2日到2031年2月3日的聊天", "2031-02-04T00:00:00+08:00")
            self.assertEqual(len(parsed["intervals"]),1)

    def test_quote_remains_separate_and_not_speaker_fact(self):
        rows = [message("A",1,"2031-02-01T00:00:00+08:00","转来参考",quote="蓝鲸项目采用西苑方案"),
                message("B",2,"2031-02-01T00:00:00+08:00","蓝鲸项目采用西苑方案",own=True)]
        db,_,_ = self.index(rows)
        with closing(db):
            hits = lookup(db,"蓝鲸项目采用西苑方案",as_of="2031-02-02T00:00:00+08:00")
            self.assertEqual(hits[0]["line"],2)
            quoted = next(h for h in hits if h["line"] == 1)
            self.assertTrue(quoted["quote_only_match"])
            self.assertEqual(quoted["body"],"转来参考")
            self.assertEqual(quoted["quoted"],rows[0]["quoted_text"])

    def test_rare_phrase_outweighs_repeated_generic_terms(self):
        rows = [message("A",1,"2031-02-01T00:00:00+08:00","蓝鲸XK72 项目的场地已确定")]
        rows += [message("B",i,"2031-02-01T00:00:00+08:00","项目场地确定项目场地确定") for i in range(2,20)]
        db,_,_ = self.index(rows)
        with closing(db):
            self.assertEqual(lookup(db,"蓝鲸XK72项目场地",as_of="2031-02-02T00:00:00+08:00")[0]["line"],1)

    def test_overlapping_events_merge_with_fixed_budget(self):
        start = datetime.fromisoformat("2031-02-01T12:00:00+08:00")
        rows = [message("A",i,(start+timedelta(seconds=i)).isoformat(),"蓝鲸计划确认 "+str(i)) for i in range(1,11)]
        rows += [message("B",1,start.isoformat(),"蓝鲸计划其他会话")]
        db,_,_ = self.index(rows)
        with closing(db):
            hits = contextual_lookup(db,"蓝鲸计划",k=2,as_of="2031-02-02T00:00:00+08:00")
            supplied = [r["line"] for h in hits for r in h["context"]]
            self.assertLessEqual(len(supplied),10)
            self.assertTrue(all(len(h["context"]) <= 5 for h in hits))
            self.assertEqual(len(supplied),len(set(supplied)))
            for h in hits:
                self.assertEqual(h["supplied_lines"],[r["line"] for r in h["context"]])
            # Results are a public JSON API: context must not refer to its hit.
            encoded = json.dumps(hits,ensure_ascii=False)
            self.assertEqual(json.loads(encoded),hits)

    def test_legacy_refused_and_existing_db_never_replaced(self):
        db,source,dest = self.index([message("A",1,"2031-01-01T00:00:00+08:00","蓝鲸")])
        db.close()
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError): build(source,dest)
        self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(),digest)
        with closing(sqlite3.connect(":memory:")) as legacy:
            legacy.execute("CREATE TABLE metadata(name TEXT,value TEXT)")
            with self.assertRaisesRegex(ValueError,"legacy"):
                lookup(legacy,"蓝鲸",as_of="2031-02-02T00:00:00+08:00")

    def test_invalid_cutoffs_and_exclusions_fail_closed(self):
        db,_,_ = self.index([message("A",1,"2031-01-01T00:00:00+08:00","蓝鲸")])
        with closing(db):
            with self.assertRaises(ValueError): lookup(db,"蓝鲸",as_of="2031-02-02T00:00:00")
            with self.assertRaises(ValueError): contextual_lookup(db,"蓝鲸",excluded_lines={"1"})
            with self.assertRaises(ValueError): lookup(db,"蓝鲸",k=0)


if __name__ == "__main__": unittest.main()

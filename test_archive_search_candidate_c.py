"""Small baseline-preserving candidate: behavior rather than benchmark labels."""
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

from experiments import archive_search_candidate_c as candidate


class CandidateCTests(unittest.TestCase):
    def test_event_date_neighbors_budget_and_visibility(self):
        rows = [dict(chat_id="A", local_id=1, time="2031-02-01T10:00:00+08:00",
                     is_self=True, text="场地怎么安排", quoted_text="")]
        rows += [dict(chat_id="B", local_id=i, time="2031-02-01T10:00:01+08:00",
                      is_self=False, text="其它会话", quoted_text="") for i in range(12)]
        rows += [dict(chat_id="A", local_id=i, time=t, is_self=False, text=b, quoted_text="")
                 for i,t,b in [(2,"2031-02-01T10:00:30+08:00","2月9日蓝鲸活动选西苑场地"),
                               (3,"2031-02-01T10:00:40+08:00","订好就通知"),
                               (4,"2031-02-01T10:00:50+08:00","蓝鲸活动此条已撤回"),
                               (5,"2031-02-06T10:00:00+08:00","蓝鲸活动未来决定")]]
        with tempfile.TemporaryDirectory() as folder:
            source, dest = Path(folder)/"source.zip", Path(folder)/"archive.sqlite"
            with zipfile.ZipFile(source,"w") as z:
                z.writestr("messages.jsonl","".join(json.dumps(r)+"\n" for r in rows))
            candidate.build(source,dest)
            with self.assertRaises(FileExistsError): candidate.build(source,dest)
            with closing(sqlite3.connect(dest)) as db:
                hits=candidate.contextual_lookup(db,"记录里提到的2月9日活动安排是什么？",
                    k=2,as_of="2031-02-05T12:00:00+08:00",excluded_lines={16})
                self.assertTrue(hits)
                self.assertEqual({r['line'] for r in hits[0]['context']},{1,14,15})
                flat=[r['line'] for h in hits for r in h['context']]
                self.assertLessEqual(len(flat),10)
                self.assertEqual(len(flat),len(set(flat)))
                self.assertNotIn(16,flat)
                self.assertNotIn(17,flat)
                for hit in hits:
                    self.assertTrue(all(abs(datetime.fromisoformat(r['ts']).timestamp()-
                        datetime.fromisoformat(hit['ts']).timestamp())<=600 for r in hit['context']))
                json.dumps(hits)


if __name__ == "__main__": unittest.main()

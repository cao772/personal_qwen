"""Semantic evidence construction tests without downloading model weights."""
from contextlib import closing
from datetime import datetime
import sqlite3
import unittest
from unittest.mock import patch
from experiments import semantic_retrieval as module

class SemanticWindows(unittest.TestCase):
    def test_noncontiguous_neighbors_exclusions_budget_and_cutoff(self):
        with closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE message(line INTEGER PRIMARY KEY,chat_id TEXT,epoch REAL,ts TEXT,is_self INTEGER,body TEXT,quoted TEXT)')
            def add(line,chat,seconds):
                db.execute('INSERT INTO message VALUES(?,?,?,?,?,?,?)',(line,chat,seconds,datetime.fromtimestamp(seconds).astimezone().isoformat(),0,'synthetic',''))
            for r in [(1,'A',100),(2,'B',110),(8,'A',120),(12,'A',130),(15,'A',140),(16,'A',150),(17,'A',800),(18,'A',900)]:add(*r)
            info={'_cutoff_epoch':850,'intervals':[]}
            with patch.object(module,'ranked_seeds',return_value=([8,17],info)):
                hits=module.contextual_lookup(db,'synthetic',k=2,excluded_lines={12})
            lines=[r['line'] for h in hits for r in h['context']]
            self.assertEqual(len(lines),len(set(lines)))
            self.assertEqual(set(lines),{1,8,15,16,17})
            for h in hits:
                self.assertLessEqual(len(h['context']),5)
                self.assertTrue(all(abs(r['epoch']-h['epoch'])<=600 for r in h['context']))
                self.assertTrue(all(r['epoch']<=850 for r in h['context']))

    def test_rrf_terms_preserve_latin_identifiers(self):
        self.assertIn('xk72',list(module.baseline_terms('蓝鲸XK72计划')))

if __name__=='__main__':unittest.main()

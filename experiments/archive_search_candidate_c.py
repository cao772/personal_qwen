"""Candidate C: baseline lexical ranking, event-date guard and real chat adjacency."""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEGMENT = re.compile(r'[\u3400-\u9fff]+|[a-z0-9]+', re.I)
NOISE = {'的是','一下','什么','多少','请问','当时','那天','我们','时候','最后','哪个','后来','记录','怎么','里面','我的','这个','那个','之前'}

def terms(text):
    for match in SEGMENT.finditer(text.lower()):
        value = match.group()
        if value.isascii() or len(value) == 1:
            yield value
        else:
            for i in range(len(value)-1):
                term = value[i:i+2]
                if term not in NOISE:
                    yield term

def build(archive, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("build a separate candidate index")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=output.parent, suffix='.sqlite')
    os.close(fd)
    try:
        with closing(sqlite3.connect(name)) as db, zipfile.ZipFile(archive) as z:
            db.executescript("""
                CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE message(line INTEGER PRIMARY KEY,chat_id TEXT,local_id TEXT,
                    ts TEXT,epoch REAL,is_self INTEGER,body TEXT,quoted TEXT);
                CREATE VIRTUAL TABLE search USING fts5(tokens,content='');
            """)
            count = 0
            with z.open('messages.jsonl') as f:
                for count, raw in enumerate(f,1):
                    msg = json.loads(raw)
                    body, quote = msg.get('text') or '', msg.get('quoted_text') or ''
                    db.execute('INSERT INTO message VALUES(?,?,?,?,?,?,?,?)',
                        (count,msg['chat_id'],str(msg['local_id']),msg['time'],
                         datetime.fromisoformat(msg['time']).timestamp(),int(msg['is_self']),body,quote))
                    db.execute('INSERT INTO search(rowid,tokens) VALUES(?,?)',
                        (count,' '.join(terms(body+' '+quote))))
            db.executemany('INSERT INTO metadata VALUES(?,?)',
                [('rows',str(count)),('zip_sha256',hashlib.sha256(Path(archive).read_bytes()).hexdigest())])
            db.commit()
        os.link(name,output)
        os.chmod(output,0o600)
    finally:
        Path(name).unlink(missing_ok=True)

def lookup(db, question, k=5, as_of=None):
    tokens = list(dict.fromkeys(terms(question)))
    if not tokens:
        return []
    if k < 1 or k > 100:
        raise ValueError('k must be 1..100')
    expression = ' OR '.join('"'+t+'"' for t in tokens[:192])
    limit_time = datetime.fromisoformat(as_of).timestamp() if as_of else 1e20
    cur = db.execute("""SELECT line,chat_id,local_id,ts,is_self,body,quoted,bm25(search) AS score
        FROM search JOIN message ON line=search.rowid WHERE search MATCH ? AND epoch<=?
        ORDER BY score,line LIMIT ?""",(expression,limit_time,k))
    cols=[c[0] for c in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]

def contextual_lookup(db, question, k=3, as_of=None, excluded_lines=()):
    """Search an explicit calendar day when present; attach nearby turns from the same chat."""
    tokens = list(dict.fromkeys(terms(question)))
    if not tokens or not 1 <= k <= 100:
        return []
    zone = timezone(timedelta(hours=8))
    cutoff = datetime.fromisoformat(as_of) if as_of else datetime.now(zone)
    if cutoff.tzinfo is None:
        raise ValueError('as_of requires a UTC offset')
    day = re.search(r'(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]', question)
    lower, upper = 0, cutoff.timestamp()
    # Preserve the baseline date rule except where the immediate noun refers
    # to an event. A remote "record/chat" word cannot turn this into send time.
    event_date = bool(day and re.match(
        r"^[的\s]*(?:活动|行程|比赛|聚餐|会议|安排|计划|预约|出差|放假|旅游|航班|车次|饭局|生日|婚礼|截止)",
        question[day.end():]))
    if day and not event_date:
        year, month, date = day.groups()
        start = datetime(int(year or cutoff.astimezone(zone).year), int(month), int(date), tzinfo=zone)
        if not year and start > cutoff:
            start = start.replace(year=start.year-1)
        lower = start.timestamp()
        upper = min(upper,(start+timedelta(days=1)).timestamp()-0.000001)
    expr = ' OR '.join('"'+t+'"' for t in tokens[:192])
    cur = db.execute("""SELECT line,chat_id,local_id,ts,is_self,body,quoted,bm25(search) AS score
        FROM search JOIN message ON line=search.rowid WHERE search MATCH ? AND epoch BETWEEN ? AND ?
        ORDER BY score,line LIMIT ?""",(expr,lower,upper,min(300,k*12)))
    cols=[c[0] for c in cur.description]
    hits=[]; used=set(excluded_lines)
    for row in cur:
        hit=dict(zip(cols,row))
        if hit['line'] in used: continue
        # Keep up to two adjacent turns per side and a ten-minute horizon.
        center=datetime.fromisoformat(hit['ts']).timestamp()
        selected = [dict(hit)]
        for before in (True, False):
            op, direction = ('<', 'DESC') if before else ('>', 'ASC')
            near=db.execute(f"""SELECT line,ts,is_self,body,quoted FROM message WHERE chat_id=?
                AND epoch BETWEEN ? AND ? AND (epoch {op} ? OR (epoch=? AND line {op} ?))
                ORDER BY epoch {direction},line {direction}""",
                (hit['chat_id'],max(lower,center-600),min(upper,center+600),center,center,hit['line']))
            count=0
            for values in near:
                if values[0] in used: continue
                selected.append(dict(zip(('line','ts','is_self','body','quoted'),values)))
                count+=1
                if count==2: break
        hit['context']=sorted(selected,key=lambda r:(datetime.fromisoformat(r['ts']).timestamp(),r['line']))
        hits.append(hit);used.update(r['line'] for r in hit['context'])
        if len(hits)==k:break
    return hits

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True);p.add_argument('--build-from');p.add_argument('--query');p.add_argument('--as-of');p.add_argument('--top',type=int,default=5)
    a=p.parse_args()
    if a.build_from:build(a.build_from,a.db)
    if a.query:
        with closing(sqlite3.connect(Path(a.db).resolve().as_uri()+'?mode=ro',uri=True)) as db:
            for hit in lookup(db,a.query,a.top,a.as_of):print(json.dumps(hit,ensure_ascii=False))

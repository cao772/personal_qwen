"""v0.6 offline lexical archive retrieval (stdlib SQLite FTS5).

Compatible entry points: build / lookup / contextual_lookup. Rebuild v0.5
indexes to a NEW path: existing databases are never overwritten or migrated in
place. This indexes original utterances, not accepted personal facts. No model,
benchmark labels, memory_store writes, or network access are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
import unicodedata
import zipfile

VERSION = "6"
ZONE = timezone(timedelta(hours=8))
NOISE = {"的是", "一下", "什么", "多少", "请问", "当时", "那天", "我们", "时候", "最后",
         "哪个", "后来", "记录", "怎么", "里面", "我的", "这个", "那个", "之前", "之后",
         "哪里", "为什么", "怎么样", "是不是", "告诉我", "有没有", "哪一个"}
DATE = re.compile(r"(?<!\d)(?:(\d{4})[年/-])?(\d{1,2})[月/-](\d{1,2})(?:日|号)?(?!\d)")
RECORD_CUE = re.compile(r"聊天|对话|消息|记录|说过|说了|问过|聊了|聊过|提到|回复|发送|那天|当天")
EVENT_CUE = re.compile(r"计划|定于|预约|预订|举办|举行|截止|到期|行程|安排")
MAX_CONTEXT_MESSAGES = 5
CONTEXT_SECONDS = 600


def normalized(text):
    return unicodedata.normalize("NFKC", text).casefold()


def terms(text):
    """Generic Han bi/trigrams and whole Latin/identifier tokens."""
    for run in re.findall(r"[\u3400-\u9fff]+|[a-z0-9_]+", normalized(text)):
        if run.isascii() or len(run) == 1:
            if run not in NOISE and run not in {"我", "你", "他", "她", "的", "了", "吗", "呢"}:
                yield run
        else:
            for n in (2, 3):
                for start in range(len(run) - n + 1):
                    token = run[start:start+n]
                    if token not in NOISE:
                        yield token


def _encoded(token):
    # Avoid unicode61 splitting identifiers/underscores differently from terms().
    return "t" + token.encode("utf-8").hex()


def _time(value):
    if not isinstance(value, str):
        raise ValueError("time must be an ISO string with a UTC offset")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("time requires a UTC offset")
    return result


def parse_query_time(question, as_of=None, *, date_from=None, date_to=None):
    """Return auditable message-time bounds, never infer future evidence.

    Bare activity dates remain lexical evidence. Dates explicitly describing
    messages and relative days are filters; explicit date_from/date_to override
    the heuristic. Calendar dates use +08:00 even if caller canonicalized as_of
    to UTC. All upper bounds are capped by as_of.
    """
    cutoff = _time(as_of) if as_of else datetime.now(timezone.utc)
    today = cutoff.astimezone(ZONE).replace(hour=0, minute=0, second=0, microsecond=0)
    result = {"as_of": cutoff.isoformat(), "calendar_timezone": "+08:00",
              "filter_applied": False, "interpretation": "as_of_only", "intervals": []}
    spans = []
    if date_from or date_to:
        start = datetime.fromisoformat(date_from).replace(tzinfo=ZONE) if date_from else None
        end = datetime.fromisoformat(date_to).replace(tzinfo=ZONE) + timedelta(days=1) if date_to else None
        for value in (date_from, date_to):
            if value and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("date_from/date_to must be YYYY-MM-DD inclusive calendar dates")
        if start and end and start >= end:
            raise ValueError("date_from is after date_to")
        spans = [(start.timestamp() if start else -1e20, end.timestamp() if end else 1e20)]
        result["interpretation"] = "explicit_message_date_filter"
    else:
        dates = list(DATE.finditer(question))
        relative = re.search(r"今天|昨天|前天|本周|这周|上周|最近\s*(\d{1,3})\s*天", question)
        record_cue = bool(RECORD_CUE.search(question))
        if dates and record_cue:
            if len(dates) > 2:
                raise ValueError("more than two calendar dates: use explicit date_from/date_to")
            starts = []
            for match in dates:
                year, month, day = match.groups()
                year = int(year) if year else today.year
                start = datetime(year, int(month), int(day), tzinfo=ZONE)
                if match.group(1) is None and start > today:
                    start = start.replace(year=year-1)
                starts.append(start)
            between = question[dates[0].end():dates[-1].start()] if len(dates) == 2 else ""
            if len(starts) == 2 and re.search(r"至|到|—|~|～", between):
                if starts[1] < starts[0]:
                    raise ValueError("query calendar range is reversed")
                spans = [(starts[0].timestamp(), (starts[1]+timedelta(days=1)).timestamp())]
            else:
                spans = [(start.timestamp(), (start+timedelta(days=1)).timestamp()) for start in starts]
            result["interpretation"] = "query_message_dates"
        elif relative and (record_cue or not EVENT_CUE.search(question)):
            label = relative.group()
            if label in {"今天", "昨天", "前天"}:
                start = today - timedelta(days={"今天": 0, "昨天": 1, "前天": 2}[label])
                end = start + timedelta(days=1)
            elif label in {"本周", "这周", "上周"}:
                start = today - timedelta(days=today.weekday() + (7 if label == "上周" else 0))
                end = start + timedelta(days=7)
            else:
                days = int(relative.group(1))
                if days < 1:
                    raise ValueError("relative day count must be positive")
                start, end = today - timedelta(days=days-1), today + timedelta(days=1)
            spans = [(start.timestamp(), end.timestamp())]
            result["interpretation"] = "relative_message_dates"
        elif dates or relative:
            result["interpretation"] = "date_mentioned_without_message_time_filter"
    result["filter_applied"] = bool(spans)
    result["intervals"] = [{"start_inclusive": low, "end_exclusive": high} for low, high in spans]
    result["_cutoff_epoch"] = cutoff.timestamp()
    return result


def _where(time_info, alias="m"):
    sql, args = f"{alias}.epoch<=?", [time_info["_cutoff_epoch"]]
    if time_info["intervals"]:
        pieces = []
        for interval in time_info["intervals"]:
            pieces.append(f"({alias}.epoch>=? AND {alias}.epoch<?)")
            args.extend([interval["start_inclusive"], interval["end_exclusive"]])
        sql += " AND (" + " OR ".join(pieces) + ")"
    return sql, args


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def build(archive, output):
    """Build a fresh v6 index; never replace an old or concurrently created DB."""
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError("index exists; build v6 at a new path, then switch explicitly")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=output.parent, suffix=".sqlite")
    os.close(fd)
    try:
        with closing(sqlite3.connect(name)) as db, ExitStack() as stack:
            db.executescript("""
                CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE message(line INTEGER PRIMARY KEY,chat_id TEXT NOT NULL,local_id TEXT NOT NULL,
                    ts TEXT NOT NULL,epoch REAL NOT NULL,is_self INTEGER NOT NULL,body TEXT,quoted TEXT,
                    chat_name TEXT,chat_type TEXT,sender TEXT,sender_id TEXT,message_type TEXT,server_id TEXT,
                    doc_len REAL NOT NULL, UNIQUE(chat_id,local_id));
                CREATE INDEX message_time ON message(epoch,line);
                CREATE INDEX message_chat_time ON message(chat_id,epoch,line);
                CREATE VIRTUAL TABLE search USING fts5(body_tokens,quote_tokens,meta_tokens,content='');
                CREATE VIRTUAL TABLE search_vocab USING fts5vocab(search,'instance');
            """)
            if zipfile.is_zipfile(archive):
                z = stack.enter_context(zipfile.ZipFile(archive))
                candidates = [x for x in z.infolist() if not x.is_dir() and
                              PurePosixPath(x.filename.replace("\\", "/")).name == "messages.jsonl"]
                if len(candidates) != 1:
                    raise ValueError("archive must contain exactly one messages.jsonl member")
                source = stack.enter_context(z.open(candidates[0]))
            else:
                source = stack.enter_context(archive.open("rb"))
            count = 0
            while raw := source.readline(8*1024*1024+1):
                if len(raw) > 8*1024*1024 or not raw.strip():
                    raise ValueError("oversized or blank source line; no silent skipping")
                count += 1
                msg = json.loads(raw)
                if type(msg.get("is_self")) is not bool or not isinstance(msg.get("chat_id"), str):
                    raise ValueError("source requires chat_id and boolean is_self")
                body, quote = msg.get("text", ""), msg.get("quoted_text", "")
                if not isinstance(body, str) or not isinstance(quote, str):
                    raise ValueError("body and quoted_text must be strings")
                ts = _time(msg["time"])
                meta = " ".join(str(msg.get(key, "")) for key in ("chat_name", "sender", "sender_id"))
                body_terms, quote_terms, meta_terms = list(terms(body)), list(terms(quote)), list(terms(meta))
                length = max(1, len(body_terms)+0.2*len(quote_terms)+0.5*len(meta_terms))
                db.execute("INSERT INTO message VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (count,msg["chat_id"],str(msg["local_id"]),msg["time"],ts.timestamp(),
                            int(msg["is_self"]),body,quote,msg.get("chat_name", ""),msg.get("chat_type", ""),
                            msg.get("sender", ""),msg.get("sender_id", ""),msg.get("type", ""),msg.get("server_id", ""),length))
                db.execute("INSERT INTO search(rowid,body_tokens,quote_tokens,meta_tokens) VALUES(?,?,?,?)",
                           (count, *[" ".join(map(_encoded, ts)) for ts in (body_terms,quote_terms,meta_terms)]))
            source_sha = _hash(archive)
            db.executemany("INSERT INTO metadata VALUES(?,?)", [
                ("schema_version",VERSION), ("rows",str(count)), ("zip_sha256",source_sha),
                ("source_sha256",source_sha), ("classification","raw_utterances_not_confirmed_facts"),
            ])
            db.commit()
        os.link(name, output)  # Atomic no-overwrite publication on the same filesystem.
        os.chmod(output, 0o600)
    finally:
        Path(name).unlink(missing_ok=True)


def _check(db):
    try:
        row = db.execute("SELECT value FROM metadata WHERE name='schema_version'").fetchone()
    except sqlite3.Error as exc:
        raise ValueError("unsupported archive index; rebuild v6 into a new path") from exc
    if not row or row[0] != VERSION:
        raise ValueError("legacy archive index; rebuild v6 into a NEW path; no in-place migration")


def _exclusions(values):
    result = set(values)
    if any(type(x) is not int or x < 1 for x in result):
        raise ValueError("excluded_lines must contain positive integer source lines")
    return result


def _dicts(cursor):
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def _rank(db, question, k, as_of, excluded_lines, date_from, date_to):
    _check(db)
    if type(k) is not int or not 1 <= k <= 100:
        raise ValueError("k must be 1..100")
    if not isinstance(question, str) or len(question) > 2048:
        raise ValueError("question must be text of at most 2048 characters")
    excluded = _exclusions(excluded_lines)
    time_info = parse_query_time(question, as_of, date_from=date_from, date_to=date_to)
    lexical = DATE.sub(" ", question) if time_info["filter_applied"] else question
    tokens = list(dict.fromkeys(terms(lexical)))
    if len(tokens) > 128:
        raise ValueError("query has more than 128 lexical terms; shorten it explicitly")
    where, args = _where(time_info)
    if not tokens:
        return [], time_info
    n, total_length = db.execute(f"SELECT count(*),coalesce(sum(doc_len),0) FROM message m WHERE {where}", args).fetchone()
    # SQL aggregates avoid loading the complete 100k-row archive into Python.
    for start in range(0, len(excluded), 400):
        chunk = sorted(excluded)[start:start+400]
        marks = ",".join("?" for _ in chunk)
        removed, length = db.execute(f"SELECT count(*),coalesce(sum(doc_len),0) FROM message m WHERE {where} AND line IN ({marks})", [*args,*chunk]).fetchone()
        n -= removed
        total_length -= length
    if not n:
        return [], time_info
    encoded = {_encoded(token): token for token in tokens}
    marks = ",".join("?" for _ in encoded)
    # FTS posting lists supply candidates AND time-visible term frequencies.
    # No BM25 corpus statistic is computed from future/excluded documents.
    postings = db.execute(f"""SELECT v.term,v.doc,v.col,count(*),m.doc_len
        FROM search_vocab v JOIN message m ON m.line=v.doc
        WHERE v.term IN ({marks}) AND {where} GROUP BY v.term,v.doc,v.col""", [*encoded,*args])
    frequencies = defaultdict(lambda: defaultdict(float))
    doc_lengths, df_docs, field_hits = {}, defaultdict(set), defaultdict(set)
    weights = {"body_tokens": 1.0, "quote_tokens": 0.2, "meta_tokens": 0.8}
    for encoded_term, line, field, frequency, length in postings:
        if line in excluded:
            continue
        token = encoded[encoded_term]
        frequencies[line][token] += frequency*weights[field]
        doc_lengths[line] = length
        df_docs[token].add(line)
        field_hits[line].add(field)
    idf = {token: math.log1p((n-len(lines)+0.5)/(len(lines)+0.5)) for token,lines in df_docs.items()}
    denominator = sum(idf.values()) or 1.0
    avg = max(total_length/n, 1.0)
    scores = {}
    for line, values in frequencies.items():
        bm25 = sum(idf[token]*2.2*tf/(tf+1.2*(0.25+0.75*doc_lengths[line]/avg)) for token,tf in values.items())
        coverage = sum(idf[token] for token in values)/denominator
        scores[line] = (bm25, coverage)
    candidate_limit = min(600, max(120, k*60))
    candidates = sorted(scores, key=lambda line: (-scores[line][0], line))[:candidate_limit]
    rows = []
    for start in range(0, len(candidates), 400):
        chunk = candidates[start:start+400]
        marks = ",".join("?" for _ in chunk)
        rows += _dicts(db.execute(f"SELECT m.* FROM message m WHERE {where} AND line IN ({marks})", [*args,*chunk]))
    # Query-derived phrases only; no personal names or benchmark terms.
    phrases = re.findall(r'[“「"]([^”」"]{2,80})[”」"]', normalized(question))
    phrases += re.findall(r"[a-z][a-z0-9_]{2,}", normalized(lexical))
    for run in re.findall(r"[\u3400-\u9fff]{4,}", normalized(lexical)):
        phrases.extend(run[i:i+4] for i in range(len(run)-3))
    phrases = list(dict.fromkeys(phrases))
    source_sha = db.execute("SELECT value FROM metadata WHERE name='zip_sha256'").fetchone()[0]
    for row in rows:
        body = normalized(row["body"])
        meta = normalized(row["chat_name"] + " " + row["sender"])
        matches = [phrase for phrase in phrases if phrase in body or phrase in meta]
        bm25, coverage = scores[row["line"]]
        phrase_bonus = min(2.0, sum(min(len(x),12)/12 for x in matches))
        rank = bm25*(0.5+coverage)+phrase_bonus
        quote_only = field_hits[row["line"]] == {"quote_tokens"}
        if quote_only:
            rank *= 0.35
        row.update({"score": -rank, "ranking_score": rank, "source_sha256": source_sha,
                    "match_fields": sorted(field_hits[row["line"]]), "quote_only_match": quote_only,
                    "ranking": {"bm25_visible": bm25, "weighted_coverage": coverage, "phrase_bonus": phrase_bonus},
                    "evidence_kind": "raw_utterance_not_confirmed_fact",
                    "quote_attribution": "quoted_content_not_automatically_speaker_experience",
                    "query_time": {key:value for key,value in time_info.items() if not key.startswith("_")}})
    return sorted(rows, key=lambda row: (row["score"],row["line"])), time_info


def lookup(db, question, k=5, as_of=None, *, excluded_lines=(), date_from=None, date_to=None):
    return _rank(db,question,k,as_of,excluded_lines,date_from,date_to)[0][:k]


def _neighbors(db, center, time_info, excluded):
    where, args = _where(time_info)
    # A copy avoids a cycle when caller later attaches this list as context.
    selected = [dict(center)]
    for before in (True, False):
        op, direction = ("<", "DESC") if before else (">", "ASC")
        cursor = db.execute(f"""SELECT m.* FROM message m WHERE {where} AND chat_id=?
            AND epoch BETWEEN ? AND ? AND (epoch {op} ? OR (epoch=? AND line {op} ?))
            ORDER BY epoch {direction},line {direction}""",
            [*args,center["chat_id"],center["epoch"]-CONTEXT_SECONDS,center["epoch"]+CONTEXT_SECONDS,
             center["epoch"],center["epoch"],center["line"]])
        names = [c[0] for c in cursor.description]
        found = 0
        for values in cursor:
            row = dict(zip(names,values))
            if row["line"] in excluded:
                continue
            selected.append(row)
            found += 1
            if found == 2:
                break
    return sorted(selected,key=lambda row:(row["epoch"],row["line"]))


def contextual_lookup(db, question, k=3, as_of=None, excluded_lines=(), *, date_from=None, date_to=None):
    """At most k event windows, each <=5 actual same-chat chronological turns."""
    excluded = _exclusions(excluded_lines)
    rows, time_info = _rank(db,question,k,as_of,excluded,date_from,date_to)
    hits = []
    used = set()
    examined = 0
    for row in rows:
        if row["line"] in used or row["line"] in excluded:
            continue
        if len(hits) >= k and not any(hit["chat_id"] == row["chat_id"] and
                abs(hit["epoch"]-row["epoch"]) <= CONTEXT_SECONDS for hit in hits):
            continue
        examined += 1
        if examined > max(12,k*8):
            break
        context = _neighbors(db,row,time_info,excluded)
        context_ids = {x["line"] for x in context}
        overlap = next((hit for hit in hits if hit["chat_id"] == row["chat_id"] and
                        context_ids & {x["line"] for x in hit["context"]}),None)
        if overlap:
            other_window_lines = {x["line"] for hit in hits if hit is not overlap
                                  for x in hit["context"]}
            union = {x["line"]:x for x in overlap["context"]+context
                     if x["line"] not in other_window_lines}
            # Merge overlapping events while retaining the strongest anchor and
            # the same fixed evidence budget; never report all seed lines as fed.
            chosen = sorted(union.values(),key=lambda x:(abs(x["epoch"]-overlap["epoch"]),x["line"]))[:MAX_CONTEXT_MESSAGES]
            if overlap["line"] not in {x["line"] for x in chosen}:
                chosen[-1] = next(x for x in union.values() if x["line"] == overlap["line"])
            overlap["context"] = sorted(chosen,key=lambda x:(x["epoch"],x["line"]))
            overlap["merged_seed_lines"].append(row["line"])
        elif len(hits) < k:
            row["context"] = context
            row["merged_seed_lines"] = [row["line"]]
            hits.append(row)
        used = {x["line"] for hit in hits for x in hit["context"]}
    for hit in hits:
        hit["supplied_lines"] = [x["line"] for x in hit["context"]]
        hit["context_budget"] = {"max_messages":MAX_CONTEXT_MESSAGES,"horizon_seconds":CONTEXT_SECONDS}
        for raw in hit["context"]:
            raw["speaker_role"] = "self" if raw["is_self"] else "other"
            raw["quote_attribution"] = "quoted_content_not_automatically_speaker_experience"
            raw["source_sha256"] = hit["source_sha256"]
    return hits


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db",required=True); p.add_argument("--build-from"); p.add_argument("--query")
    p.add_argument("--as-of"); p.add_argument("--top",type=int,default=5)
    p.add_argument("--date-from"); p.add_argument("--date-to"); p.add_argument("--context",action="store_true")
    a = p.parse_args()
    if a.build_from: build(a.build_from,a.db)
    if a.query:
        with closing(sqlite3.connect(Path(a.db).resolve().as_uri()+"?mode=ro",uri=True)) as db:
            fn = contextual_lookup if a.context else lookup
            for hit in fn(db,a.query,a.top,a.as_of,date_from=a.date_from,date_to=a.date_to):
                print(json.dumps(hit,ensure_ascii=False))

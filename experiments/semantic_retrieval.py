"""Frozen experiment D: local BGE message embeddings and lexical/dense RRF.

Private arrays never leave the execution host. This module uses original message
bodies (quotes explicitly labelled) only, not future neighbors or benchmark gold.
Date bounds come from candidate B; no learned time/person resolution is claimed.
"""
from __future__ import annotations
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from experiments.archive_search_candidate_b import parse_query_time, _where, _time, _exclusions

INSTRUCTION = '为这个句子生成表示以用于检索相关文章：'
RRF_CONSTANT = 60
CANDIDATE_LIMIT = 100
MODEL_FILES = ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']
STATE = None


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def model_fingerprints(path):
    return {name:sha256(Path(path)/name) for name in MODEL_FILES if (Path(path)/name).exists()}


def encoder(model_path):
    import torch
    from transformers import AutoModel,AutoTokenizer
    if torch.cuda.device_count()!=1: raise RuntimeError('Bind one GPU explicitly')
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModel.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
    return tok,model


def encode(texts,tok,model):
    import torch
    x=tok(texts,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        v=model(**x).last_hidden_state[:,0].float()
        return torch.nn.functional.normalize(v,p=2,dim=1)


def build(db_path,model_path,output,batch_size=64):
    import numpy as np
    import torch
    out=Path(output);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    started=time.perf_counter()
    tok,model=encoder(model_path)
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri()+'?mode=ro',uri=True)) as db:
        rows=db.execute('SELECT line,epoch,is_self,body,quoted FROM message ORDER BY line').fetchall()
        source_sha=db.execute("SELECT value FROM metadata WHERE name='zip_sha256'").fetchone()[0]
    vectors=np.lib.format.open_memmap(out/'vectors.npy',mode='w+',dtype='float32',shape=(len(rows),model.config.hidden_size))
    lines=np.array([r[0] for r in rows],dtype=np.int64)
    epochs=np.array([r[1] for r in rows],dtype=np.float64)
    np.save(out/'lines.npy',lines);np.save(out/'epochs.npy',epochs)
    truncated=0
    for start in range(0,len(rows),batch_size):
        texts=[('本人：' if r[2] else '他人：')+(r[3] or '')+('\n引用（不自动归于发送者）：'+r[4] if r[4] else '') for r in rows[start:start+batch_size]]
        lengths=tok(texts,truncation=False,return_length=True)['length']
        truncated+=sum(n>512 for n in lengths)
        vectors[start:start+len(texts)]=encode(texts,tok,model).cpu().numpy()
        if start%6400==0:print(json.dumps({'embedded':min(start+batch_size,len(rows)),'total':len(rows)}),flush=True)
    vectors.flush()
    manifest={'model':'BAAI/bge-small-zh-v1.5','model_files':model_fingerprints(model_path),
              'source_archive_sha256':source_sha,'database_sha256':sha256(db_path),'rows':len(rows),
              'dimensions':int(model.config.hidden_size),'pooling':'normalized CLS','precision':'FP16 forward, FP32 normalized vectors',
              'max_length':512,'truncated_documents':truncated,'seconds':time.perf_counter()-started,
              'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
              'arrays':{n:sha256(out/n) for n in ['vectors.npy','lines.npy','epochs.npy']},
              'implementation_sha256':sha256(__file__)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


def state(db):
    global STATE
    import numpy as np
    import torch
    folder=Path(os.environ['PQ_SEMANTIC_INDEX'])
    model_path=Path(os.environ['PQ_EMBED_MODEL'])
    current=db.execute("SELECT value FROM metadata WHERE name='zip_sha256'").fetchone()[0]
    if STATE is None:
        manifest=json.loads((folder/'manifest.json').read_text())
        if current!=manifest['source_archive_sha256']:raise ValueError('Wrong source archive')
        for name,h in manifest['arrays'].items():
            if sha256(folder/name)!=h:raise ValueError('Corrupted vector index')
        if model_fingerprints(model_path)!=manifest['model_files']:raise ValueError('Embedding model changed')
        tok,model=encoder(model_path)
        lines=np.load(folder/'lines.npy',allow_pickle=False)
        epochs=np.load(folder/'epochs.npy',allow_pickle=False)
        actual=db.execute('SELECT line,epoch FROM message ORDER BY line').fetchall()
        if actual!=list(zip(lines.tolist(),epochs.tolist())):raise ValueError('Index row mapping changed')
        matrix=torch.from_numpy(np.load(folder/'vectors.npy',allow_pickle=False)).to('cuda')
        STATE={'folder':folder,'model_path':model_path,'source':current,'tok':tok,'model':model,
               'lines':lines,'epochs':epochs,'vectors':matrix}
    if STATE['source']!=current or STATE['folder']!=folder or STATE['model_path']!=model_path:
        raise ValueError('Do not change index or model within one process')
    return STATE


def baseline_terms(text):
    noise={'的是','一下','什么','多少','请问','当时','那天','我们','时候','最后','哪个','后来','记录','怎么','里面','我的','这个','那个','之前'}
    for value in re.findall(r'[\u3400-\u9fff]+|[a-z0-9]+',text.lower()):
        if value.isascii() or len(value)==1:yield value
        else:
            for i in range(len(value)-1):
                if value[i:i+2] not in noise:yield value[i:i+2]


def ranked_seeds(db,question,as_of,excluded,mode):
    import numpy as np
    import torch
    s=state(db)
    info=parse_query_time(question,as_of)
    valid=s['epochs']<=info['_cutoff_epoch']
    if info['intervals']:
        intervals=np.zeros(len(valid),dtype=bool)
        for t in info['intervals']:intervals|=(s['epochs']>=t['start_inclusive'])&(s['epochs']<t['end_exclusive'])
        valid&=intervals
    valid&=~np.isin(s['lines'],list(excluded))
    q=encode([INSTRUCTION+question],s['tok'],s['model'])[0]
    scores=(s['vectors']@q).masked_fill(~torch.from_numpy(valid).to('cuda'),float('-inf'))
    n=min(CANDIDATE_LIMIT,int(valid.sum()))
    if not n:return [],info
    # Stable line ordering gives deterministic tie-breaking.
    ids=torch.argsort(scores,descending=True,stable=True)[:n].cpu().tolist()
    dense=[int(s['lines'][i]) for i in ids]
    if mode=='dense':return dense,info
    where,args=_where(info)
    if excluded:
        where+=' AND m.line NOT IN ('+','.join('?' for _ in excluded)+')';args+=sorted(excluded)
    tokens=list(dict.fromkeys(baseline_terms(question)))[:192]
    lexical=[]
    if tokens:
        expression=' OR '.join('"'+t+'"' for t in tokens)
        lexical=[r[0] for r in db.execute(f'''SELECT m.line FROM search JOIN message m ON m.line=search.rowid
             WHERE search MATCH ? AND {where} ORDER BY bm25(search),m.line LIMIT ?''', [expression,*args,CANDIDATE_LIMIT])]
    fusion={}
    for ranking in (lexical,dense):
        for rank,line in enumerate(ranking,1):fusion[line]=fusion.get(line,0)+1/(RRF_CONSTANT+rank)
    return sorted(fusion,key=lambda line:(-fusion[line],line)),info


def contextual_lookup(db,question,k=2,as_of=None,excluded_lines=()):
    if type(k) is not int or not 1<=k<=100:raise ValueError('k must be 1..100')
    if not isinstance(question,str) or len(question)>2048:raise ValueError('Invalid query')
    excluded=_exclusions(excluded_lines)
    mode=os.environ.get('PQ_SEMANTIC_MODE','hybrid')
    if mode not in ('dense','hybrid'):raise ValueError('Unknown semantic mode')
    seeds,info=ranked_seeds(db,question,as_of,excluded,mode)
    where,args=_where(info)
    used=set(excluded);hits=[]
    for line in seeds:
        if line in used:continue
        cur=db.execute('SELECT * FROM message WHERE line=?',(line,)); names=[r[0] for r in cur.description]
        center=dict(zip(names,cur.fetchone()))
        selected=[dict(center)]
        for before in (True,False):
            op,direction=('<','DESC') if before else ('>','ASC')
            cur=db.execute(f'''SELECT m.* FROM message m WHERE {where} AND chat_id=?
                AND epoch BETWEEN ? AND ? AND (epoch {op} ? OR (epoch=? AND line {op} ?))
                ORDER BY epoch {direction},line {direction}''',
                [*args,center['chat_id'],center['epoch']-600,center['epoch']+600,center['epoch'],center['epoch'],line])
            names=[r[0] for r in cur.description];count=0
            for values in cur:
                row=dict(zip(names,values))
                if row['line'] in used:continue
                selected.append(row);count+=1
                if count==2:break
        center['context']=sorted(selected,key=lambda r:(r['epoch'],r['line']))
        center['retrieval_mode']=mode
        hits.append(center);used.update(r['line'] for r in selected)
        if len(hits)==k:break
    return hits


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();build(a.db,a.model,a.out)

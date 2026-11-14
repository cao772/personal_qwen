# 独立的时间、窗口、语义检索与证据阅读实验

这些文件是研究候选，不改变现有入口。冻结的 A 保持原字节；B、C 和语义方案各自保存结果，不将事后修复套入 A 的历史成绩。

- `experiments/archive_search_candidate_b.py`：从 A 分叉，只把消息日期线索限定到日期后紧邻的谓语，并限制合并窗口内所有消息距主锚点不超过 600 秒。需要 schema 6 索引。
- `experiments/archive_search_candidate_c.py`：保留旧单列 FTS5 的双字查询和 BM25 排序，添加活动日期名词保护及同会话时间邻接。使用旧索引结构。日期规则仍有歧义；不是完整语义时间解析器。
- `experiments/semantic_retrieval.py`：在受控服务器上为每条原文计算独立向量，比较纯语义与词法/语义合并。原文不会发送给模型 API；只读取预先下载并核验的本地模型。
- `experiments/evidence_reader.py`：保留原回答提示及新增核对提示，用于控制证据截断和阅读行为。后者由已见失败启发，必须报告为开发实验。

## 语义方案的固定设置

底座为 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)，版本 `7999e1d3359715c523056ef9478215996d62a620`。按官方模型卡使用归一化 CLS，查询加检索指令，原文不加检索指令。最长 512 token，FP16 前向后转 FP32 归一化；构建报告记录截断数量。

向量只来自单条消息的作者角色、正文及显式标注的引用，不包含后续消息或问题标签。检索时按 `as_of`、排除行、B 的日期规则过滤，再取最多 100 个语义候选。合并模式对两个最多 100 项的排名采用 `sum(1/(60+rank))`，常数和上限在评测前固定。最多返回 k 个窗口，各最多五条消息；窗口按同一会话真实时间邻接构建，距中心前后各不超过 600 秒。

**限制**：C 和语义合并的词法部分保留 SQLite FTS5 的全库 BM25 统计。未来/排除消息不会作为返回证据，但可能影响词法排名统计。因此它们不满足 A/B 的“未来文档不改变分数”性质。密集向量的时间过滤不解决跨群人物别名、引用归属推断或活动日期歧义。相似度和融合分数都不是事实置信度。

## 独立构建与复现

先准备与旧单列 FTS5 兼容的私有索引，并按模型版本下载配置、tokenizer、safetensors 权重。核验模型仓库的文件大小与 Git blob/LFS 哈希。以下路径均为占位符；不要把个人原文、索引或输出提交到公开仓库。

```bash
# 从仓库根目录执行，并只选择已检查空闲的一个 GPU。
python -m experiments.semantic_retrieval \
  --db /private/archive.sqlite \
  --model /private/bge-small-zh-v1.5 \
  --out /private/new-semantic-index

export PQ_EMBED_MODEL=/private/bge-small-zh-v1.5
export PQ_SEMANTIC_INDEX=/private/new-semantic-index
export PQ_SEMANTIC_MODE=hybrid  # 或 dense
python evaluate_retrieval.py \
  --module experiments/semantic_retrieval.py \
  --db /private/archive.sqlite \
  --queries /private/frozen-queries.jsonl \
  --archive-sha256 REPLACE_WITH_SOURCE_SHA256 \
  --out /private/new-result
```

依赖 PyTorch、Transformers、NumPy；不要求新增向量数据库。建库输出新目录，包含数组和哈希清单。加载时核验数组、模型文件、源 ZIP 标识及索引行号/时间对应关系。时间解释的依赖文件也应单独记录哈希。默认评估器的首题耗时包含模型/索引加载；比较稳定延迟时应先用合成问题预热，并如实注明排除了启动成本。

对照生成时，所有条件保持底座、生成参数和证据格式相同。先比较同样截断的检索结果与标注原文，再单独比较完整标注原文，最后才改变提示。标注原文条件是诊断条件，不能作为日用系统成绩；命中一条原文也不能保证截断后保留了答案。

## 合成检查

```bash
python -B -m unittest -v \
  test_archive_search_candidate_a test_evaluate_retrieval \
  test_archive_retrieval_known_limits test_archive_search_candidate_b \
  test_archive_retrieval_fixes test_archive_search_candidate_c \
  test_semantic_retrieval
```

冻结 A 的两个已知反例仍为预期失败；B 上相同反例必须成功。语义上下文的合成检查不下载权重，真实模型构建、数组有限性、单位范数、逐题检索和生成另行在私有执行环境核验。

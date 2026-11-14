# 冻结检索实验与来源评估工具

本次提供 `experiments/archive_search_candidate_a.py` 和 `evaluate_retrieval.py`。它在本机索引已提供的文字归档，返回可回查的消息。它不加载模型，不连接聊天账号，不训练权重，也不把聊天内容自动登记为已确认的个人事实。

**候选 A 未通过效果验收，作为负结果冻结保存。现有日用系统继续使用已验证版本。** 下述建库与查询命令用于复现实验。公开仓库只包含通用代码与合成测试，真实题目、原文、索引和逐题结果应保存在私有目录。检索结果仍可能包含个人信息；不要把 CLI 的 JSON 输出粘贴到公开 issue、PR 或运行日志。

## 建立独立索引

需要 Python 3.10 或更新版本，以及启用 FTS5 的 SQLite。仅使用 Python 标准库；不安装模型或下载依赖。

```bash
python experiments/archive_search_candidate_a.py \
  --build-from /path/to/private/export.zip \
  --db /path/to/private/archive-v06.sqlite
```

也可以提供 `messages.jsonl` 文件。ZIP 内必须恰有一个 basename 为 `messages.jsonl` 的成员。消息需保留 `chat_id`、`local_id`、带时区的 `time`、布尔型 `is_self`、字符串正文和引用；可选的会话名称及发送者字段进入独立元数据列。空正文仍保留，重复会话内消息标识、空行或超过 8 MiB 的单行会报错。

索引元数据保存完整输入文件 SHA256 与 schema 版本 6。为兼容已有调用方，`zip_sha256` 字段继续存在；直接输入 JSONL 时它表示该 JSONL 的 SHA256。新文件权限为 `600`，新建目录为 `700`。输入文件不修改。

**目标索引必须不存在。** 旧索引会明确提示重建，不进行原地迁移。切换已有助手时，先在独立目录完成验证；停下旧客户端后再成对切换代码与索引，并保留两者的旧版本供回退。只替换 Python 文件而保留旧索引，会产生明确的 schema 错误。

## 查询与证据预算

以下问句和日期均为合成示例：

```bash
python experiments/archive_search_candidate_a.py \
  --db /path/to/private/archive-v06.sqlite \
  --query '昨天群里说园艺活动需要带什么？' \
  --as-of '2025-03-10T20:00:00+08:00' \
  --context --top 2
```

Python 入口继续提供：

```python
build(archive, output)
lookup(db, question, k=5, as_of=None)
contextual_lookup(db, question, k=3, as_of=None, excluded_lines=())
```

`lookup` 现在也接受 keyword-only `excluded_lines`。两个查询函数都可用 `date_from`、`date_to` 明确限制消息日期。`score` 仍为数值越小排名越靠前，不能当作可信度概率。

每个上下文窗口最多 5 条消息，中心前后各最多 2 条，限制在前后 10 分钟内。邻接按同一会话的时间和来源行号确定，不依赖消息在整个归档中恰好相邻。窗口重叠时合并，最终不同窗口不重复提供同一条消息。`k=2` 最多提供 10 条。

保留 `line/chat_id/local_id/ts/is_self/body/quoted/score/context` 等旧字段；新增会话名称、发送者、原始来源 SHA256、排序解释、时间解释和 `supplied_lines`。`supplied_lines` 才表示最终提供的证据。`merged_seed_lines` 只说明哪些候选中心参与过合并，不能拿它统计供给证据。

`is_self` 只表示原始导出的发言角色。引用独立保存在 `quoted` 中；引用命中会降低排序权重，并标注不能自动当作当前发送者的亲历。发送者名称不等于已经解决跨群、别名或同名人物的身份归并。

## 时间与可见性

`as_of` 是带 UTC 偏移的 ISO 时间，始终限制候选、邻接消息与排序统计；不提供时取当前时间。`excluded_lines` 中的来源行也从这些环节排除。排除某行并不等于删除了其它消息对同一事件的转述。

自然时间采用有限规则，日历时区固定为 `+08:00`，以兼容调用方把时间规范化为 UTC 的情况。明确谈论消息、聊天、回复的日期可形成过滤；单纯提及活动日期时通常保留为词法线索。日期与活动时间混用仍可能有歧义。需要确定范围时使用显式参数，起止日期均包含：

```bash
python experiments/archive_search_candidate_a.py \
  --db /path/to/private/archive-v06.sqlite \
  --query '园艺活动需要带什么？' \
  --date-from 2025-03-08 --date-to 2025-03-09 \
  --as-of '2025-03-10T20:00:00+08:00' \
  --context --top 2
```

没有年份的日期使用不晚于截止日的最近年度；复杂跨年范围应使用带年份的显式参数。问题超过 2048 字符或产生超过 128 个词项会报错，不静默截断。当前接口以内容检索为用途，不提供完整的按日期浏览功能。

## 检索方法与边界

中文使用双字和三字片段，Latin、数字和标识符使用完整词项。FTS5 倒排供给词项频率，BM25 使用时间可见且未被排除的文档计算。候选再依据词项覆盖、问题中的短语及正文、引用、会话元数据等通用信号排序。

这是词法方法。没有共同词项的同义表达仍可能漏检；窗口跨度不足也可能遗漏较早的问句或后续决定。排序分数不证明事实正确，找到原文也不证明生成模型能正确理解人物、否定或前后版本。

它与版本化记忆库是不同职责：版本化记忆负责已登记事实、纠正和撤回；原文检索负责找到依据。候选来源不会自动写进记忆库。

## 冻结小测并比较

先由未看新实现或结果的人编写、核验并冻结查询文件，再比较两版。算法只接收问题、截止时间和排除行；金标及参考答案留在评估器。已经看过结果后用于调参的集合应改称开发或回归集。

私有 JSONL 每行采用以下格式，内容为合成示例：

```json
{"id":"synthetic-01","question":"园艺活动要带什么？","as_of":"2025-03-10T20:00:00+08:00","answerable":true,"gold_source_lines":[12,13],"excluded_lines":[]}
```

未知题使用 `answerable=false`，缺少的答案不能因为检索到了相关活动就算正确。确实有关的上下文可另记 `related_source_lines`；它不进入正题的来源覆盖分母。

```bash
python evaluate_retrieval.py \
  --module /path/to/trusted/archive_search_candidate_a.py \
  --db /path/to/private/archive-v06.sqlite \
  --queries /path/to/private/frozen-queries.jsonl \
  --archive-sha256 REPLACE_WITH_FROZEN_SOURCE_SHA256 \
  --out /path/to/private/new-evaluation-directory
```

`--module` 会导入该 Python 文件，只应指向已审阅的本地实现。来源哈希必须来自冻结数据集的来源清单，并与索引相同。输出目录必须新建。评估器核验金标行存在、满足截止时间，并把每条返回消息的时间、正文和引用与索引原行核对。输出记录实现、评估器、查询及来源的 SHA256。

默认比较前 2 个窗口、最多 10 条不同消息、每条正文 120 字符和引用 60 字符。相同上限不意味着实际长度或模型 token 数完全相同，因此同时记录实际可见字符量。检索延迟只计 `contextual_lookup` 调用，不包含生成和来源核验。

报告应分开列出：

1. 可答题中找齐全部标注来源的数量，以及至少命中一条的数量。
2. 来源行覆盖数量与实际供给长度；金标可能比回答最低所需更严格。
3. 模型最终答对率、未知题是否克制、人物与版本归属错误。这部分必须另接真实模型测量，不能由检索覆盖推算。

截断后所需答案可能不再可见，来源行命中不是答案正确率。研究者编制题也不能代表日常自然问题的分布。

## 合成验证

```bash
python -B -m unittest -v test_archive_search_candidate_a test_evaluate_retrieval test_archive_retrieval_known_limits
```

测试覆盖时间与排除边界、真实会话邻接、窗口预算与去重、JSON 序列化、未来文档统计隔离、旧索引不被覆盖，以及评估器的来源完整性、未知题分母和金标不进入检索参数。

## 冻结候选的已知问题

代码保留初次效果评估时的字节，不把事后修复混入原结果。两个合成反例以 `expectedFailure` 保留在 `test_archive_retrieval_known_limits.py`：

- 日期误过滤：截至某月 5 日询问“之前记录里提到的该月 9 日活动”，可能错误过滤至消息发送日 9 日，甚至回退上一年；真正依据可能是 1 日发出的活动计划。活动日期与消息日期混用时，当前启发式不足。
- 窗口跨度：重叠窗口合并后，保留的消息可能超出主锚点前后 600 秒，尽管元数据仍声明该范围。消息数预算与 `as_of` 仍受限，但跨度声明不完全成立。

因此上述时间消歧和窗口跨度是设计意图及已知边界，不能解读为通过验证的完整保证。所有初始合成测试通过也不能代替效果评估。后续应以新候选、新测试和单独结果推进，保留此冻结版本供比较。

# 固定来源的证据供给与阅读诊断

这些工具用于分开检查三件事：检索是否找到依据、输入是否仍包含必要原句、模型是否正确理解。来源行出现不等于所需片段可见；片段可见也不等于回答正确。当前打包器保留为研究组件，尚未通过日用效果验收。

## 保持问题和来源不变

`prepare_evidence_cases.py` 使用冻结的查询文件及既有检索轨迹，不重新排序或检索。每条轨迹必须含 `id`、问题原文字节的 `question_sha256` 和 `retrieved_source_lines`。同一 ID 换了问题文字也会被拒绝。

程序先核验完整源 ZIP 的 SHA256，再按全局行号读取原始消息及发送者字段。时间晚于问题截止、处于排除列表、来源不存在或角色格式不明确时直接报错。参考答案、可答标签和金标都不会传给打包器。

```bash
python prepare_evidence_cases.py \
  --archive /path/to/private/original-export.zip \
  --archive-sha256 REPLACE_WITH_FROZEN_SOURCE_SHA256 \
  --queries /path/to/private/frozen-queries.jsonl \
  --retrieval /path/to/private/frozen-retrieval.jsonl \
  --out /path/to/private/new-packing-comparison
```

不指定 `--max-chars` 时，每题预算严格等于其原固定前缀格式的实际证据字符数，包含原来的标签和换行。原控制保留每条正文 120 字符、引用 60 字符的处理方式。输出 `legacy_cases.jsonl`、`packed_cases.jsonl`、`packing_audit.jsonl` 和哈希清单。

使用 `--max-chars 2400` 可单独检查较宽总预算。它与等预算条件必须分开报告，不能把增加篇幅带来的改善归因于选句规则。相同字符上限也不等于相同模型 token 数。

如果来源行通过答案标注选择，须明确添加 `--source-kind oracle`，并在轨迹中标记该类型。这个条件用于阅读诊断，不能报告为检索效果，也不是每道题最终准确率的严格上界；删除原先上下文可能改变理解。

输出目录须新建，使用私有文件权限。真实输入、来源作者、片段审计及模型回答应留在仓库之外。

## 打包器接口与原文审计

```python
from experiments.evidence_packing import pack_evidence

packed = pack_evidence(
    question,
    verified_messages,
    archive_sha256=source_hash,
    max_chars=character_budget,
    max_messages=10,
)
case["correct"] = packed["evidence"]
```

输入消息必须显式提供 `line/ts/is_self/body/quoted`；空正文或引用用空字符串表示。可提供源档案中的 `sender/sender_id/chat_id/chat_name`。打包器不猜姓名、事实主体或引用原作者，不继承相邻锚点的发送者。它不负责来源读取或时间过滤；直接使用 API 时，调用方必须先完成这些检查。

模型文本使用短来源号、消息时间、本人/他人和发送者显示名，正文与引用单独呈现。元数据及原句经过 JSON 字符串转义；省略处明确显示标记。完整档案哈希、发送者 ID、原文字段偏移留在旁侧审计中，不反复放进模型输入。

规则优先保留能容纳的完整短消息；长消息依据问题词项选完整句及邻句，无命中时采用明确标记的首尾片段。选句仍可能漏掉同义表述、跨消息关系或省略部分中的纠正，不能把片段当成完整事实。

`packed["rendered_text"]` 严格等于 `"\n".join(packed["evidence"])`。总预算包含所有头部、转义、引用、省略标记和连接换行。很小的预算可能连完整标签与正文都无法容纳，此时返回空供给及原因，而不是隐藏额外开销。

每个原文字段的审计包含：

- 原文字数和 SHA256。
- 保留的 `[start,end)` Unicode 字符区间及精确原文。
- 省略区间、是否完整，以及包含档案哈希和原文行号的展开引用。
- 未供给整条消息时的原因；该消息仍保留回查信息。

保留区间与省略区间应完整覆盖原字段。坐标不是 UTF-8 字节或模型 token 坐标，也不保证片段语义完整。

## 阅读运行的指纹与长度记录

`experiments/evidence_reader.py` 仍接收 `id/question/correct`，两个 research 提示常量、chat 提示以及生成参数保持原语义。`prompt_version` 现在逐次显式传入，避免在同一 Python 进程里修改全局提示状态。

每条实际模型输出增加以下审计字段：

| 字段 | 含义 |
| --- | --- |
| `input_tokens` | 实际模板编码得到的输入 token 数 |
| `evidence_sha256` | 实际证据字符串数组的规范 JSON 指纹 |
| `chat_messages_sha256` | 传入聊天模板的 role/content 消息序列指纹 |
| `input_token_ids_sha256` | 实际输入 token ID 序列指纹，可发现模板或分词变化 |
| `requested_prompt_version` / `effective_prompt_version` | 请求与实际使用的提示版本 |
| `case_file_sha256` | 原始输入文件字节指纹；stdin 为 null |
| `max_input_tokens` / `max_new_tokens` | 本次配置的输入与输出限制 |

证据和消息数组使用规范 UTF-8 JSON，保留字符串内的空格；token 指纹取自唯一一次实际模板编码，在生成计时之前计算。

chat 模式使用原 chat 提示，即使请求 checked，日志也如实记为 `effective_prompt_version=chat`。这次没有把 checked 指令偷偷加到 chat 模式。

```bash
python experiments/evidence_reader.py \
  --model /path/to/existing-local-model \
  --cases /path/to/private/packed_cases.jsonl \
  --out /path/to/private/new-answers.jsonl \
  --prompt-version original --mode research \
  --max-input-tokens 4096 --max-new-tokens 192
```

使用现有共享设备预检与单 GPU 运行入口执行。模型及 tokenizer 只从已存在的本地目录加载。`--max-input-tokens` 是可选的明确限制；超过时停止并报错，不截断证据。省略该选项维持原行为。这段命令说明用法，不表示已经执行模型验证。

## 如何记录结果

对同一来源集合比较原前缀、同来源全文、新格式；先把来源可见性与最终回答分开。标注必要片段时，应保留否定、猜测、计划与多子问要求，并允许经过核实的同义来源。精确金标行未命中，并不能自动证明没有其它有效证据。

报告逐题改善和退步，总分相同可能隐藏互换。未知题单列，空供给不自动算拒答正确。根据已知失败补写的提示或片段标签属于开发诊断，需要另取未参与修改的材料验收。

现有开发诊断已显示，额外标签与引用表达可能挤占很小的字符预算，因此此格式不作为日用默认。扩大预算、补全来源或新增作者信息各改变了输入条件，应分别记录。代码提供来源与运行核验能力，不承诺回答质量提升。

```bash
python -B -m unittest -v test_evidence_packing test_prepare_evidence_cases test_evidence_reader_audit
```

这些是纯 CPU 合成检查，覆盖来源绑定、问题指纹、时间可见性、原文偏移、预算、提示状态与 token 审计路径；不是实际 Qwen 推理结果。

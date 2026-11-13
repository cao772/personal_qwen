# 个人 Qwen：全生活记忆、个性化与结构裁剪实验

版本：0.3，2026-11-13。

## 项目范围

研究对象是长期服务一个人的私人模型。范围包括生活经历、日常习惯、兴趣、关系与人物、性格表达、情绪与交流方式、学习、工作以及跨场景的全部已提供聊天记录。任何一个工作项目都不能代表整个个人。

所有来源先完整保留，再生成可追溯的记忆和训练样本。原始消息、助手回复、用户纠正、分支版本、角色扮演和后来改变的决定必须保留区别。单次表达不自动变成永久人格，助手提出的建议也不自动变成用户已经做过的事。具体规则见 [全生活数据设计](docs/whole_life_memory.md)。

代码仓库只包含通用程序、研究设计和标注为合成的示例。真实聊天、记忆库、训练样本、服务器身份、运行日志与私人适配器保存在仓库外的私有目录。

## 本轮交付状态

这是可检查的离线研究代码，不包含预训练模型权重，也不包含已经训练好的私人模型。

已实现：Qwen3-1.7B 官方结构核对、参数预算、dense Qwen3 FFN 物理裁剪工具、目标任务校准入口、最小可训练神经记忆模块与训练入口、保留原始来源的聊天导入器、共享 GPU 机器的预检与运行脚本，以及合成数据格式示例。已运行的验证与尚未运行的部分分别记录在 reports/validation.json。

未完成：下载或加载真实 Qwen 权重、运行真实 Qwen 推理、执行任务校准、剪后恢复训练、个人 LoRA、训练记忆模块、真实个人数据评测与部署。训练代码只做语法和不依赖 PyTorch 的入口检查。实际验证结果在 reports/validation.json。

当前执行环境有 NumPy，没有 PyTorch、Transformers 或可用 NVIDIA 训练环境。代码未安装依赖或自动下载模型。

## 目标与不可混淆的概念

目标是让较小模型在生活与工作中可靠地使用历史、适应交流方式、接受纠正和续办事情。权重没有现成的“数学知识”“闲聊知识”分区，本包不能按知识主题无损删除权重。校准分数反映的是有限样本下的近似贡献，不证明某个通道在未来永远无用。

结构化裁剪会真的减小矩阵；稀疏置零不一定减少普通稠密推理耗时；量化减少每个参数所占字节；蒸馏训练较小模型保留目标行为。四者应分别测量。

## 选定的首个结构基线

Qwen/Qwen3-1.7B，标准纯文本 dense Qwen3。官方配置：28 层、隐藏维度 2048、FFN 中间维度 6144、16 个 Q 头/8 个 KV 头、共享输入/输出词表权重。

| 候选 | FFN 中间维度 | 按配置计算的总参数 | 总参数减少 |
|---|---:|---:|---:|
| 原始结构 | 6144 | 1,720,574,976 | 0% |
| 候选 A | 5376 | 1,588,454,400 | 约 7.68% |
| 候选 B | 4608 | 1,456,333,824 | 约 15.36% |

这是结构预算，不是能力保留率、速度实测或已生成模型。精确推导由 architecture_budget.py 给出。最合适裁剪比例需由留出任务表现与实际运行成本决定。

首版只裁 FFN 中间通道。同一组通道索引同时作用于 gate_proj/up_proj 的输出行和 down_proj 的输入列；保持层数、隐藏维度、注意力头、词表和归一化维度不变。

## 文件说明

| 文件 | 用途 | 当前验证 |
|---|---|---|
| architecture_budget.py | 不加载权重，按标准配置计算参数预算 | 已执行 |
| qwen_ffn_prune.py | 检查、计划、实际写出更小的 dense 权重 | 合成矩阵与文件测试 |
| calibrate_ffn.py | 用本地任务文本估计 FFN 通道贡献 | 语法/CLI；未前向 |
| memory_prefix.py | 查询条件化的神经记忆读取器、输入前缀拼接 | 语法；未张量前向 |
| train_memory_prefix.py | 冻结底座，训练记忆侧模块 | 语法/CLI/数据划分检查；未训练 |
| import_chat_history.py | 原样保留导出文件，索引全部消息与分支 | 合成导出测试；未导入真实历史 |
| import_tracememo.py | 导入 TraceMemo 微信文字 JSONL/ZIP，区分本人、其他发言人和引用 | 合成测试；私有来源验证另存 |
| scripts/ | 共享服务器只读预检与 GPU1 运行入口 | 离线检查；未连真实服务器 |
| docs/whole_life_memory.md | 全生活数据范围、来源与时间规则、评测设计 | 设计；完整数据库尚未实现 |
| docs/source_coverage.md | ChatGPT/Codex/微信来源覆盖差异与本机导出交接 | 来源说明与操作任务 |
| experiment_plan.json | 候选结构、对照组、未完成项 | 设计 |
| data/ | 工作与生活的合成格式示例 | 不是个人训练集 |
| reports/ | 参数预算和实际验证记录 | 区分计算/测试/未运行 |
| SOURCES.md | 官方来源与方法关系 | 本轮已核对 |

## 运行准备

使用独立 Python 环境。仅裁剪所需依赖见 requirements-prune.txt；校准/记忆训练所需依赖见 requirements-train.txt。依赖范围是研究起点，本轮未安装测试完整组合，正式训练须记录实际环境版本。

需要自行提供本地 Hugging Face 模型目录，包含 config.json、model.safetensors 和 tokenizer 文件。裁剪与校准只支持单个未量化 model.safetensors、dense Qwen3、F32/F16/BF16。分片、GGUF、GPTQ/AWQ、MoE、Qwen3.5 混合架构不在本工具范围内，会报错而非尝试随意切片。

以下 /path/to/... 是需要替换的本地路径，不会触发网络下载。裁剪结果总是写入全新目录，不修改来源。

### 1. 不需要模型权重的检查

```bash
python architecture_budget.py
python -m unittest discover -v
```

### 1a. 导入已取得的聊天导出

```bash
python import_chat_history.py --input /path/to/chatgpt-export.zip --output-dir /path/to/private/new-import
```

也接受直接提供 conversations.json。输出目录必须不存在，父目录须已存在。输出 source_original.zip/json、messages.jsonl 和 manifest.json，目录权限 700、文件权限 600。默认 JSON 读取上限 256 MiB，可用 --max-json-bytes 显式调整；超过上限报错，不截断。

ZIP 保留整包但不解包附件，只读取唯一的 conversations.json；重名候选会拒绝。索引包括所有 mapping 节点和非当前分支，保留原始角色、时间、内容与来源指针。完整性仅针对这个输入文件，不能据此宣称整个账号已经全部导出。训练器不直接接受原始消息索引，需要后续整理为有来源和时间边界的样本。

### 1b. 导入 TraceMemo 微信文字记录

```bash
python import_tracememo.py --input /path/to/wechat-text-export.zip --output-dir /path/to/private/new-wechat-import
```

也接受直接提供 messages.jsonl；ZIP 中必须恰有一个该文件。按行处理，保留原始输入及哈希，输出 records.jsonl 和 manifest.json。目录必须新建，权限为 700/600；默认总 JSONL 上限 1 GiB、单行 8 MiB、最多 100 万条，可显式调整对应参数。超限、重复消息标识或格式错误会报告失败，不静默截断或去重。

字段包括 chat_id/chat_name/chat_type、sender/sender_id、is_self、带时区的 time、local_id/server_id、type/text，以及引用消息的 quoted_text。完整 parsed raw_record 和未知字段都保留，精确原字节以源文件为准。输出通过 is_self 区分 self/other；对方不是 assistant，引用也不能直接当作发送者本人的事实。空正文和空显示名继续保留并计数，不凭显示名猜身份。

此导入结果不是 SFT 训练集。它没有提炼人格、决定哪些话应被模仿，或把原文写进模型权重。ChatGPT 的可见记忆、相关历史检索、账号导出、Codex 本地会话和生成记忆也分别登记，详见 [来源覆盖与交接](docs/source_coverage.md)。

### 2. 检查真实本地模型

```bash
python qwen_ffn_prune.py inspect --model-dir /path/to/Qwen3-1.7B
```

### 3. 运行目标任务校准

校准 JSONL 每行格式为 `{"text":"有来源的任务与正确处理过程"}`。覆盖生活问答、偏好变化、人物关系、情绪交流、学习与工作，同时保留中文、否定、时间、代码和指令理解等基础样本。用验证集决定保留能力，不把短期不活跃视为永远无用。

```bash
python calibrate_ffn.py --model-dir /path/to/Qwen3-1.7B --data-jsonl /path/to/calibration.jsonl --output /path/to/scores.json --device cuda --dtype bfloat16 --max-tokens 2048
```

评分近似为 FFN 中间通道激活的均方值乘下投影对应列权重的平方范数。它忽略通道交互等因素，必须实测裁剪损失。分数文件绑定配置与完整源权重 SHA256，避免被误用于同结构的另一模型。提供分数时，plan/crop 会流式读取源权重校验身份。data/calibration_demo.jsonl 仅为合成格式示例，不是实际校准数据。

### 4. 生成计划，再写出候选权重

```bash
python qwen_ffn_prune.py plan --model-dir /path/to/Qwen3-1.7B --scores-json /path/to/scores.json --target-intermediate-size 5376
python qwen_ffn_prune.py crop --model-dir /path/to/Qwen3-1.7B --scores-json /path/to/scores.json --target-intermediate-size 5376 --output-dir /path/to/Qwen3-personal-candidate --apply
```

省略 --apply 时 crop 只输出计划，不写文件。省略 --scores-json 时按权重幅值做演示排序；这不能当作按个人任务识别重要性的结果。

### 5. 剪后恢复与选择底座

**此步骤没有在本包中实现。** 对裁剪候选做蒸馏或微调恢复，并与原始模型、原生更小模型对照。只有候选在个人留出任务和基本语言任务上满足要求，才选为记忆模块底座。不能跳过这一步就宣称私人模型质量不变。

也可以先在未裁剪底座上验证记忆模块。最终裁剪版确定后，需要重新对齐或重训记忆模块，不能直接默认兼容。

### 6. 训练最小记忆模块

输入为 JSONL 的 question/answer/memories。训练器提供 chronological 与 manual 两种明确划分声明。下例的合成数据只用于运行管线，不能用来宣称实际记忆能力：

```bash
python train_memory_prefix.py --model-dir /path/to/selected-local-Qwen --train-jsonl data/train_demo.jsonl --eval-jsonl data/eval_demo.jsonl --output-dir /path/to/new-memory-run --split-declaration chronological --device cuda --num-slots 8 --reader-dim 128 --epochs 1
```

默认 8 个记忆槽、读取器维度 128。底座参数冻结，答案损失经过底座反向传到记忆读取器。问题和记忆前缀不作为预测目标。回答不进入记忆编码，避免直接泄露答案。

为了降低首版依赖和成本，编码器仅使用底座输入词向量的均值池化；它会丢词序与细节，不是成熟的句向量模型或 Memorilla 复现。所有候选记忆由调用者显式提供，不暗中 top-k 丢弃；数量或 token 超限会报错，不静默截断。

输出 memory_prefix.pt、memory_config.json、history.jsonl。只报告教师强制条件下的答案 loss，不报告召回率或生成质量。重载时必须核验底座配置及全部权重指纹；本包尚未提供完整推理服务或自动重载验证器。

## 如何真正加入个人信息

1. 原始聊天：通过 import_chat_history.py 导入本人提供的 ChatGPT JSON 或 ZIP，通过 import_tracememo.py 导入 TraceMemo 微信文字记录。保存原文件副本和哈希，保留消息身份和来源；文本索引不替代多模态内容或原始档案。其它聊天平台需增加对应格式适配器，目前没有自动账号全量导出能力。
2. 稳定表达和协作方式：整理“请求—错误回答—纠正—最终接受版本”以及适用场景，用于个人适配。日常闲聊、倾诉、学习和工作分别标注，个人 LoRA 在本包中仅为后续设计。
3. 新经历、人物、关系和动态进度：进入可更新、可追溯的源记录；问题发生后才出现的信息不能进入该问题的历史。不要从单次倾诉推定永久性格。
4. 记忆使用能力：先学习从记录中找答案、处理更新/撤销、人物归属和证据不足，再用个人任务适配。少量个人记录无法保证从零训练出成熟记忆机制。

本仓库没有真实用户历史，也没有从聊天摘要重建或猜测原始对话。完整导出、记忆提取、可更新索引、训练数据审核、个人适配和生成式评测均需在私有数据环境继续完成。导入成功只说明该输入文件被保留，不能证明账号的所有历史已经导出。

## 正式验收设计

对照完整底座、恢复后的裁剪底座、完整底座+记忆、裁剪底座+重新对齐记忆。个人适配另做有/无对照。

重点测：训练后新增事实、偏好随时间改变、一次性情绪与长期风格、角色扮演与本人经历、亲友与本人信息归属、随机编号与姓名、旧决定被撤销、相似场景串线、当前状态续办、有效约束遗漏、无依据编造。再做移除/打乱记忆对照，证明模型实际使用了正确记忆。

分别记录生成答案正确率、必要证据召回、延迟、峰值设备内存及整套系统存储。底座参数少不等于系统更省资源；软记忆槽也不意味着无限无损记忆。

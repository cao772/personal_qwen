# 合成格式实验的运行核验与盲判包

`audit_format_run.py` 只使用 Python 标准库。它核对已冻结的合成输入与现有 reader 输出，并准备人工判读文件。它不运行模型、不加载 tokenizer、不训练，也不自动评判答案正确性。

集成时把 `audit_format_run.py` 和 `test_format_run_audit.py` 放到仓库根目录；测试使用仓库里的 `experiments.format_controls` 和 `experiments.evidence_reader`。冻结 reader 源码保持原样。

## 使用

在实验输入冻结时，另行保存 `manifest.json` 原始字节的 SHA256。运行核验时提供这个已经记录的值，不要临时把待核验 manifest 的新指纹当成可信根。

```bash
python audit_format_run.py \
  --bundle frozen_format_bundle \
  --manifest-sha256 FROZEN_MANIFEST_SHA256 \
  --answers reader_answers.jsonl \
  --reader experiments/evidence_reader.py \
  --out new_private_review
```

工具要求一个全新输出目录；发现错误会退出失败，不跳过坏行。成功后生成目录权限 `700`、文件权限 `600` 的三个文件：

- `request_audit.json`：包文件、完整唯一 ID、实际证据和消息哈希、提示版本、输入上限和输出 cap 的核验记录与限制。
- `blind_cases.jsonl`：新盲判 ID、问题、原始证据、原始答案、是否可回答、冻结的参考答案和判读要求，以及输出 token 数与 `hit_cap`。只把这个文件交给判读者。
- `blind_key.json`：盲判 ID 与原 case ID、场景和格式条件的映射，另存保管。

乱序由固定种子和原 case ID 的规范 JSON SHA256 决定，输入输出行的顺序变化不改变盲判顺序。可以用 `--shuffle-seed` 显式指定并记录另一个种子。条件标签隐藏，但证据中的格式仍然可见；这不构成完全遮蔽格式条件的实验。

## 核验范围

外部固定的 manifest 指纹、manifest 文件清单中的全部文件指纹和字节数、完整唯一 case ID 集、`condition=correct`、案例文件原字节 SHA256、证据数组规范 JSON SHA256 和 chat messages 规范 JSON SHA256 都必须一致。输入 case 只接受 `id/question/correct`；运行记录的未知字段、缺失字段、重复 JSON 键、空行和错误记录都拒绝。数量按 manifest 声明核对，适用于 136 条格式矩阵和独立的 24 条桥接输入，不限制某一个 version 字符串。

manifest 的说明性元数据由外部 SHA256 固定，不逐项解释。条件映射和 contrast 只核对文件指纹、来源绑定与 ID 对应，不重复生成器的全部因子设计检查；桥接包附带的 `pack_audit.jsonl` 核对完整文件指纹和字节数，不重跑 packer 或逐段审核 spans。

工具只支持已审阅的冻结 reader SHA256 `f61cd69be33495206257a7afa743dec45c35fbe1051596504e46e1e0dbd7fda2`。实际 reader 文件和 manifest 必须同时匹配这个值，之后只调用其纯 CPU `chat_messages`。请求固定为 `research/original`、`max_new_tokens=192`、`max_input_tokens=4096`；证据与消息不做截断或改写。

`input_tokens` 必须在 1–4096 内，输出 `tokens` 必须在 0–192 内；非空答案不能记录为 0 个 token。`hit_cap` 必须与 `tokens == 192` 完全一致。合法的 `hit_cap=true` 保留给人工判断，不自动判错。相同答案出现在不同 case 中是允许的；相同 case ID 出现两次会拒绝。

没有实际 tokenizer 和 token ID 序列，因此 `input_token_ids_sha256` 只检查指纹格式，不重新计算；输入及输出 token 数只检查范围和字段一致性。审计记录明确标注这些限制。冻结代码中的提示和生成设置、输出所记录的指纹只能验证请求记录的自洽性，不能证明记录来自真实模型运行。当前 reader 不独立记录模型或 adapter 身份，所以工具也不宣称核验了实际模型身份。运行设备、实际权重、环境与解码执行也未经独立验证，需要执行端另存相应记录。

## 合同测试

集成到仓库后运行：

```bash
python -m unittest test_format_run_audit.py
```

测试使用生成器的合成输入与人工构造的模拟输出，覆盖正确绑定、确定性乱序、权限、缺失/重复/多余 ID、证据或提示指纹变动、reader 与 manifest 变动、cap 矛盾、未知/缺失字段、坏行和文件内容改动。模拟 token 数和模拟答案都不是模型结果；测试通过不代表 Qwen 已运行或答对。

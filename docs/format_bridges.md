# 冻结格式的合成桥接

受控矩阵中的额外边界本身可能改变阅读，因此另准备三组桥接，直接调用现有冻结格式组件。原 136 条矩阵和文件保持不动；桥接使用相同八个虚构情景，每个情景三组，共 **24 条输入**。两套合计 160 条，仍然只有八个情景。

| 组 | 呈现方式 |
| --- | --- |
| A | 原 `prepare_evidence_cases.legacy_evidence`，不显示作者名字 |
| B | 原 `experiments.evidence_packing.pack_evidence`，完整供给、普通合成显示名 |
| C | 与 B 相同，仅将预定主体的显示名换成 `assistant` |

**B/C 只改变显示名。A/B 同时改变多项格式，属于复合桥接，不能归因于某一个字段。** 源正文、引用、顺序、角色、时间、问题和判分标准均与原矩阵一致。显示名不作为 API 角色，也不改变 `is_self`。

生成器按仓库包路径导入既有组件，在调用格式函数前核验原矩阵生成器、实际导入的旧格式准备器与证据打包器、声明的组件文件以及读取器的源码哈希，不重写这些组件。所有正文与引用均短于原 120/60 字符限制，且空白归一化不会改变它们；任何不满足项直接报错。B/C 给足篇幅，并逐条核验供给消息数、全文位置、原句哈希、`complete`、无省略区间及实际呈现字符串。源记录没有被截断。

打包器 API 的 `archive_sha256` 在这里绑定的是合成来源清单，没有真实聊天 ZIP。`source_catalog.json` 与 136 条矩阵的对应清单字节一致。记录编号是合成情景中的唯一派生编号。

## 生成及执行

```bash
python -m experiments.format_bridge_controls \
  --out /path/to/new-synthetic-bridge \
  --legacy-preparer prepare_evidence_cases.py \
  --packer-file experiments/evidence_packing.py \
  --reader-file experiments/evidence_reader.py
```

输出 `cases.jsonl`、独立 `answer_key.jsonl`、`condition_map.jsonl`、`contrasts.jsonl`、`source_catalog.json`、`pack_audit.jsonl` 和 `manifest.json`。其中只有 `cases.jsonl` 交给模型。`pack_audit.jsonl` 保存 B/C 共 16 条原打包器的完整审计结果。

沿用原始本地底座、原提示、greedy、关闭 thinking、4096 输入上限及 192 输出上限，使用已有单 GPU 预检入口。读取器运行命令与矩阵相同，仅更换 cases 和新的输出路径。生成器不加载模型，也不生成回答。

## 判读

预声明的八组 A/B 与八组 B/C 都应报告，不挑选成功配对。继续使用原八个情景的判分标准，单列内容正确性、主体错误、无依据内容及仅复制元数据。矩阵与桥接分开列条件成绩。

此桥接检查格式效应在原呈现实现上是否出现，不能构成真实个人记忆或新生活材料的验收。即使桥接未复现某种错误，也不能据此证明旧真实输入没有该错误。本人／他人碰撞分别落在不同情景，分组差异只作描述。显示名变化同时改变字符串及 token，不能推断模型内部机制。

当前状态为 `prepared_not_executed`，没有实际模型输入 token 数、回答或效果分数。

## 合同检查

```bash
python -B -m unittest -v test_format_bridge_controls
```

测试直接验证调用现有格式函数、组件身份、完整原句、B/C 只有名字变化、旧截断约束、文件哈希及可重复性。独立工具目录测试时可用 `FORMAT_TEST_COMPONENT_ROOT` 指向包含原公共组件的仓库根目录。

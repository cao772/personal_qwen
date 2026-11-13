# 公开依据与可复现状态

检索日期：2026-11-13。以下均为模型作者、实现维护方或论文作者的一手材料。

1. Qwen3-1.7B 模型卡： https://huggingface.co/Qwen/Qwen3-1.7B
2. 本轮读取的 Qwen3-1.7B 配置： https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json
3. Qwen3 实现（MLP 三矩阵、GQA、Q/K RMSNorm）： https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
4. NVIDIA Model Optimizer 裁剪官方说明，含 Qwen3-8B 示例与剪后恢复要求： https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/pruning/README.md
5. Wanda 论文，权重与激活重要性、稀疏与结构裁剪的区别： https://arxiv.org/abs/2306.11695
6. Memorilla 论文与项目： https://arxiv.org/abs/2610.21375 ， https://memory.stanford.edu/
7. Memorilla 官方训练工程： https://github.com/snap-stanford/memorilla
8. LoRA： https://arxiv.org/abs/2106.09685

## 本包与这些工作的关系

- 本包是独立的最小研究原型，不是 NVIDIA Minitron、Wanda 或 Memorilla 的官方复现。
- 结构裁剪只覆盖标准 dense Qwen3 的 FFN 中间通道；不会声称已识别某种知识主题对应的权重。
- 本包的幅值排序仅为演示；激活能量评分也只是有限校准任务上的贡献近似，不能保证未来任务不受影响。
- 官方 Memorilla 的已发布检查点绑定 Qwen3-8B 解码器及 Qwen3-Embedding-4B 编码器；不能直接认为适配本包 1.7B 或裁剪版。
- 官方完整三阶段训练配方的硬件成本不代表本包的最低要求，也不构成本包训练速度或效果承诺。
- reference config 来自上述动态 main URL，尚未下载并锁定真实模型 revision；正式训练应记录实际 checkpoint revision/文件校验值与依赖版本。
- 结构参数预算为根据配置与标准实现推导的算术结果；不是读取了预训练权重后的测量，也不是吞吐量实测。

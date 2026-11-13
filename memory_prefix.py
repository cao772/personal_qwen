"""Experimental, trainable soft-memory prefix for a frozen causal LM.

This is NOT a reproduction of Memorilla. The accompanying trainer uses mean
pooled input-token embeddings, a deliberately weak and lossy baseline encoder.
Neither fixed-size prefixes nor this encoder preserve every historical detail.
Keep source records separately. An untrained reader is not a working memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MemoryPrefixConfig:
    memory_dim: int
    query_dim: int
    llm_hidden_size: int
    reader_dim: int = 128
    num_slots: int = 8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class MemoryPrefixReader(nn.Module):
    """Read all supplied candidate memories with question-conditioned slots.

    Inputs: query [B, Q], memories [B, M, D], boolean valid mask [B, M].
    Output: soft prefix [B, S, H]. No hidden top-k selection is performed.
    Empty/all-masked memory rows attend to a learned null-memory value instead
    of applying softmax to an all-negative-infinity row.
    """

    def __init__(self, config: MemoryPrefixConfig) -> None:
        super().__init__()
        self.config = config
        d = config.reader_dim
        self.slots = nn.Parameter(torch.randn(config.num_slots, d) * 0.02)
        self.null_value = nn.Parameter(torch.zeros(d))
        self.query_projection = nn.Linear(config.query_dim, d, bias=False)
        self.key_projection = nn.Linear(config.memory_dim, d, bias=False)
        self.value_projection = nn.Linear(config.memory_dim, d, bias=False)
        self.query_norm = nn.LayerNorm(d)
        self.read_norm = nn.LayerNorm(d)
        self.output_projection = nn.Linear(d, config.llm_hidden_size, bias=False)

    def forward(
        self, query: Tensor, memories: Tensor, memory_mask: Tensor
    ) -> Tensor:
        cfg = self.config
        if query.ndim != 2 or query.shape[-1] != cfg.query_dim:
            raise ValueError("query must have shape [batch, query_dim]")
        if memories.ndim != 3 or memories.shape[-1] != cfg.memory_dim:
            raise ValueError("memories must have shape [batch, count, memory_dim]")
        if query.shape[0] != memories.shape[0]:
            raise ValueError("query and memories batch sizes differ")
        if memory_mask.shape != memories.shape[:2] or memory_mask.dtype != torch.bool:
            raise ValueError("memory_mask must be boolean [batch, memory_count]")
        if query.device != self.slots.device or memories.device != self.slots.device:
            raise ValueError("reader and input embeddings must be on the same device")
        if memory_mask.device != memories.device:
            raise ValueError("memory_mask and memories must be on the same device")

        query = query.to(dtype=self.slots.dtype)
        memories = memories.to(dtype=self.slots.dtype)
        queries = self.query_norm(
            self.slots.unsqueeze(0) + self.query_projection(query).unsqueeze(1)
        )
        keys = self.key_projection(memories)
        values = self.value_projection(memories)
        batch_size = query.shape[0]
        # The null entry is accessible only for an empty/all-masked row.
        null_key = keys.new_zeros((batch_size, 1, cfg.reader_dim))
        null_value = self.null_value.view(1, 1, -1).expand(batch_size, 1, -1)
        keys = torch.cat((keys, null_key), dim=1)
        values = torch.cat((values, null_value), dim=1)
        valid = torch.cat((memory_mask, ~memory_mask.any(dim=1, keepdim=True)), dim=1)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(cfg.reader_dim)
        scores = scores.float().masked_fill(~valid.unsqueeze(1), float("-inf"))
        attention = torch.softmax(scores, dim=-1).to(values.dtype)
        read = torch.matmul(attention, values)
        return self.output_projection(self.read_norm(queries + read))


def freeze_base_model(model: nn.Module) -> None:
    """Freeze parameters, not the forward graph needed by the prefix reader."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def prepend_memory_prefix(
    token_embeddings: Tensor,
    token_attention_mask: Tensor,
    token_labels: Tensor,
    prefix: Tensor,
) -> dict[str, Tensor]:
    """Build an HF causal-LM input; caller must already mask prompt labels.

    Prefix labels are always -100. This function preserves gradients from the
    decoder's answer loss into `prefix`, including when dtypes differ.
    """
    if token_embeddings.ndim != 3 or prefix.ndim != 3:
        raise ValueError("token_embeddings and prefix must both be rank 3")
    batch, length, hidden = token_embeddings.shape
    if prefix.shape[0] != batch or prefix.shape[2] != hidden:
        raise ValueError("prefix batch/hidden dimensions must match token embeddings")
    if token_attention_mask.shape != (batch, length):
        raise ValueError("attention mask has the wrong shape")
    if token_labels.shape != (batch, length) or token_labels.dtype != torch.long:
        raise ValueError("labels must be int64 [batch, token_count]")
    if token_attention_mask.device != token_embeddings.device or token_labels.device != token_embeddings.device:
        raise ValueError("token inputs must be on one device")
    prefix = prefix.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    prefix_length = prefix.shape[1]
    prefix_mask = token_attention_mask.new_ones((batch, prefix_length))
    attention_mask = torch.cat((prefix_mask, token_attention_mask), dim=1)
    labels = torch.cat(
        (token_labels.new_full((batch, prefix_length), -100), token_labels), dim=1
    )
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "inputs_embeds": torch.cat((prefix, token_embeddings), dim=1),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
    }

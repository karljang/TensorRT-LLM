# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorRT-LLM PyTorch backend models for Qwen3-TTS.

Qwen3-TTS is a TTS pipeline with two transformer models:
  - Talker: 28-layer GQA Transformer with M-RoPE and QK-norm
    (generates codebook-0 semantic tokens from text)
  - CodePredictor: 5-layer GQA Transformer with standard RoPE and QK-norm
    (generates codebooks 1-15 from codebook-0 and talker hidden states)

The Talker architecture is similar to Qwen3-VL (M-RoPE + QK-norm + GQA).
The CodePredictor is a standard Qwen3-like transformer.

Note: These models are registered separately from the standard LLM pipeline
because Qwen3-TTS requires a custom nested generation loop where the
CodePredictor runs inside each Talker decode step.
"""

from typing import Optional

import torch
from torch import nn

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.qk_norm_attention import QKNormRoPEAttention
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model

# ---------------------------------------------------------------------------
# Talker model (main 28-layer transformer with M-RoPE)
# ---------------------------------------------------------------------------


class Qwen3TTSTalkerAttention(QKNormRoPEAttention):
    """Talker attention with M-RoPE and QK-norm."""

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: Optional[int] = None,
    ) -> None:
        config = model_config.pretrained_config

        # M-RoPE configuration (same pattern as Qwen3-VL)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            pos_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            mrope_section = rope_scaling.get("mrope_section", None)
            mrope_interleaved = rope_scaling.get("interleaved", False)
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.from_string("mrope" if mrope_section else pos_type),
                rope=RopeParams.from_config(config),
                mrope_section=mrope_section,
                mrope_interleaved=mrope_interleaved,
            )
            fuse_qk_norm_rope = not mrope_interleaved
        else:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )
            fuse_qk_norm_rope = True

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=getattr(config, "attention_bias", False),
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=getattr(config, "attention_bias", False),
            config=model_config,
        )


class Qwen3TTSTalkerDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config

        self.self_attn = Qwen3TTSTalkerAttention(model_config, layer_idx=layer_idx)

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        mrope_config: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            mrope_config=mrope_config,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)

        return hidden_states, residual


class Qwen3TTSTalkerModel(DecoderModel):
    """Talker backbone: 28-layer GQA Transformer with M-RoPE.

    Has two embedding tables:
      - codec_embedding (vocab=3072): for codec/speech tokens
      - text_embedding (vocab=151936): for text tokens (projected via text_projection)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        config = model_config.pretrained_config

        # Codec embedding (used as the primary input_embedding)
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        # Text embedding (separate, projected via text_projection)
        self.text_embedding = nn.Embedding(
            getattr(config, "text_vocab_size", 151936),
            getattr(config, "text_hidden_size", config.hidden_size),
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3TTSTalkerDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        mrope_config: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at "
                "the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mrope_config=mrope_config,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Qwen3TTSTalkerForCausalLM")
class Qwen3TTSTalkerForCausalLM(DecoderModelForCausalLM[Qwen3TTSTalkerModel, object]):
    """Talker model with codec LM head.

    Registered for potential use with TRT-LLM's LLM API for the
    autoregressive generation of codebook-0 tokens.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(
            Qwen3TTSTalkerModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )


# ---------------------------------------------------------------------------
# CodePredictor model (5-layer transformer, generates codebooks 1-15)
# ---------------------------------------------------------------------------


class Qwen3TTSCodePredictorAttention(QKNormRoPEAttention):
    """CodePredictor attention with standard RoPE and QK-norm."""

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: Optional[int] = None,
    ) -> None:
        config = model_config.pretrained_config

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(config),
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=getattr(config, "attention_bias", False),
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=True,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=getattr(config, "attention_bias", False),
            config=model_config,
        )


class Qwen3TTSCodePredictorDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config

        self.self_attn = Qwen3TTSCodePredictorAttention(model_config, layer_idx=layer_idx)

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)

        return hidden_states, residual


class Qwen3TTSCodePredictorModel(DecoderModel):
    """CodePredictor backbone: 5-layer GQA Transformer."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        config = model_config.pretrained_config
        num_code_groups = getattr(config, "num_code_groups", 16)

        # Per-codebook embeddings (groups 1-15)
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.torch_dtype)
                for _ in range(num_code_groups - 1)
            ]
        )

        self.layers = nn.ModuleList(
            [
                Qwen3TTSCodePredictorDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            raise ValueError("CodePredictor requires inputs_embeds")

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


__all__ = [
    "Qwen3TTSTalkerAttention",
    "Qwen3TTSTalkerDecoderLayer",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSTalkerForCausalLM",
    "Qwen3TTSCodePredictorAttention",
    "Qwen3TTSCodePredictorDecoderLayer",
    "Qwen3TTSCodePredictorModel",
]

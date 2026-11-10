# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The MiniMax-H3 pipeline of the ``minimax_h3_nvfp4__sm_107__u1_u4`` scenario.

The family pipeline with the scenario's transformer (:mod:`.modeling`) and the work outside the denoise
loop as the recipe runs it: the VAE ViT decoder's Linears as FP8 per-tensor GEMMs with the tiled and
fused quantize path, the decoder's spatial tiles, the decoder call replayed from CUDA graphs per chunk
shape, and the text-only text-encoder prefill replayed from a CUDA graph per prompt length. All of it
is selected by :mod:`.config`; the family's own path (``visual_gen_scenario=off``) is untouched.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.logger import logger

from ..packing import MINIMAX_H3_TEXT_ENCODER_LAYER
from ..pipeline_minimax_h3 import MiniMaxH3Pipeline
from .config import SCENARIO
from .kernels.vae_decoder_fp8 import convert_decoder_linears_fp8, set_tiles
from .modeling import ScenarioTransformer


class MiniMaxH3Nvfp4Sm107U1U4Pipeline(MiniMaxH3Pipeline):
    """MiniMax-H3 on R200 (SM 107): pre-quantized NVFP4 Linears, FP8 trtllm-gen attention, Ulysses 1 or 4."""

    @classmethod
    def resolve_variant(cls, config):  # the resolver has already chosen this class
        return cls

    def _init_transformer(self) -> None:
        self.transformer = ScenarioTransformer(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        )

    def post_load_weights(self) -> None:
        # The family sets the schedulers' shifts and prepares the decoder (``_prepare_vae_decoder``);
        # the scenario adds its decoder tiles, the FP8 decoder Linears and the decoder graphs.
        if self.vae is not None and self.device.type == "cuda":
            tile = SCENARIO.vae_tile
            if tile:
                # Before the decoder calibration, which measures one tile of this geometry.
                self.vae.enable_tiling(
                    tile_sample_min_height=tile[0],
                    tile_sample_min_width=tile[1],
                    tile_sample_min_overlap_height=tile[2],
                    tile_sample_min_overlap_width=tile[2],
                )
                logger.info(f"MiniMax-H3 VAE decoder tiles {tile[0]}x{tile[1]} px, min overlap {tile[2]}")
            self.vae.decoder_graph_max_shapes = int(SCENARIO.vae_graph_max_shapes)
        super().post_load_weights()
        if self.vae is not None and self.device.type == "cuda" and SCENARIO.vae_fp8:
            ulysses = self._ulysses_size()
            if SCENARIO.vae_fp8_tiled:
                set_tiles(mlp=SCENARIO.vae_fp8_tile, q=SCENARIO.resolved_vae_fp8_qtile(ulysses))
            convert_decoder_linears_fp8(self.vae.decoder)

    def _ulysses_size(self) -> int:
        vgm = self.pipeline_config.visual_gen_mapping
        return int(getattr(vgm, "ulysses_size", 1) or 1) if vgm is not None else 1

    # ------------------------------------------------------------------ text encoder CUDA graph
    def _text_prefill_fast(
        self, input_ids: torch.Tensor, vision_inputs: dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not SCENARIO.text_encoder_graph or vision_inputs or self.device.type != "cuda":
            return None
        with self.offloader.context_if_requested("text_encoder"):
            return self._encode_text_tokens_graph(input_ids)

    def _text_prefill_layers(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The text-only Qwen3-VL prefill up to the layer MiniMax-H3 reads, as the modules run it.

        Replicates ``Qwen3VLModel.forward`` -> ``Qwen3VLTextModel.forward`` for a text-only prompt: the
        token embeddings, the 4-row position ids (text row + the three mRoPE rows, all ``arange``),
        the rotary tables, then the decoder layers with no attention mask (the HF call's all-ones mask
        is dropped by its ``_ignore_causal_mask_sdpa`` check, so the layers run with ``attention_mask=None``
        and SDPA's ``is_causal`` path either way). ``hidden_states[k]`` of the HF output is the output of
        layer ``k - 1``, so the first ``MINIMAX_H3_TEXT_ENCODER_LAYER`` layers are run and the later ones,
        which the HF call also executes, are skipped. No host synchronisation, so the call captures.
        """
        text_model = self.text_encoder.model.language_model
        hidden_states = text_model.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, hidden_states.shape[0], -1)
        text_position_ids, mrope_position_ids = position_ids[0], position_ids[1:]
        position_embeddings = text_model.rotary_emb(hidden_states, mrope_position_ids)
        for decoder_layer in text_model.layers[:MINIMAX_H3_TEXT_ENCODER_LAYER]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    def _encode_text_tokens_graph(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``_text_prefill_layers`` replayed from a CUDA graph keyed by the prompt's token count.

        The first prompt of a length runs the layers eagerly once (the same kernels warm), then
        captures; later prompts of that length copy their ids into the static input and replay.
        The static output is cloned, since the next replay of this length overwrites it. At most
        ``text_encoder_graph_max`` lengths are kept (LRU).
        """
        seq_len = int(input_ids.shape[1])
        graphs = self.__dict__.setdefault("_te_graphs", {})
        entry = graphs.get(seq_len)
        if entry is None:
            if len(graphs) >= SCENARIO.text_encoder_graph_max:
                oldest = next(iter(graphs))
                del graphs[oldest]
            static_ids = input_ids.clone()
            self._text_prefill_layers(static_ids)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._text_prefill_layers(static_ids)
            entry = (graph, static_ids, static_out)
            logger.info(
                f"MiniMax-H3 text encoder: CUDA graph captured for {seq_len} tokens ({len(graphs) + 1} lengths)"
            )
        else:
            del graphs[seq_len]  # re-insert: dict order is the LRU order
        graphs[seq_len] = entry
        graph, static_ids, static_out = entry
        static_ids.copy_(input_ids)
        graph.replay()
        return static_out.clone().to(device=self.device, dtype=self.dtype)

#!/usr/bin/env python
"""CLIP text encoder and prompt learner for supervised UCF101 training.

The learnable text branch is:

    learnable context + action class tokens + EOT
    -> frozen CLIP text Transformer
    -> ln_final
    -> token_features [101, 77, 512]
    -> Text_MSTFF in VitaCLIP_model.py
    -> text_projection
    -> class text features [101, 512]

The fixed-template branch uses the ordinary ``forward`` method and remains
frozen. Both branches share this same CLIP text encoder.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from VitaCLIP_text_encoder_utils import SimpleTokenizer as _Tokenizer
from Text_MSTFF import TextMSTFF


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """LayerNorm that performs its calculation in FP32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        output = super().forward(x.float())
        return output.to(dtype=original_dtype)


def tokenize(
    texts: Union[str, Sequence[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> torch.LongTensor:
    """Tokenize one string or a sequence of strings with CLIP BPE.

    Returns:
        Tensor with shape ``[num_texts, context_length]`` and dtype long.
    """
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    tokenizer = _Tokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    all_tokens = [
        [sot_token] + tokenizer.encode(text) + [eot_token]
        for text in texts
    ]

    result = torch.zeros(
        len(all_tokens),
        context_length,
        dtype=torch.long,
    )

    for index, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(
                    f"Input {texts[index]!r} is too long for "
                    f"context_length={context_length}."
                )
            tokens = tokens[:context_length]
            tokens[-1] = eot_token

        result[index, : len(tokens)] = torch.tensor(
            tokens,
            dtype=torch.long,
        )

    return result


class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

        if attn_mask is None:
            self.register_buffer(
                "attn_mask",
                None,
                persistent=False,
            )
        else:
            self.register_buffer(
                "attn_mask",
                attn_mask,
                persistent=False,
            )

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        attention_mask = None
        if self.attn_mask is not None:
            attention_mask = self.attn_mask.to(
                device=x.device,
                dtype=x.dtype,
            )

        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attention_mask,
        )[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.width = int(width)
        self.layers = int(layers)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=self.width,
                    n_head=heads,
                    attn_mask=attn_mask,
                )
                for _ in range(self.layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        maple_prompts: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run the frozen CLIP text Transformer.

        Args:
            x: Text tokens in sequence-first layout ``[L, N, D]``.
            maple_prompts: Optional old MaPLe deep prompt sequence retained
                for backward compatibility.
        """
        use_maple = maple_prompts is not None and len(maple_prompts) > 0

        if not use_maple:
            for block in self.resblocks:
                x = block(x)
            return x

        num_prompts = int(maple_prompts[0].shape[0])

        if len(maple_prompts) < max(0, self.layers - 1):
            raise ValueError(
                "Not enough maple_prompts for the text Transformer: "
                f"need at least {self.layers - 1}, got {len(maple_prompts)}."
            )

        for layer_index, block in enumerate(self.resblocks):
            if layer_index > 0:
                prefix = x[:1, :, :]
                suffix = x[1 + num_prompts :, :, :]

                textual_context = maple_prompts[layer_index - 1]
                textual_context = textual_context.to(
                    device=x.device,
                    dtype=x.dtype,
                )
                textual_context = textual_context.unsqueeze(1).expand(
                    -1,
                    x.shape[1],
                    -1,
                )

                x = torch.cat(
                    [prefix, textual_context, suffix],
                    dim=0,
                )

            x = block(x)

        return x


class CLIPTextEncoder(nn.Module):

    def __init__(
        self,
        embed_dim: int = 512,
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        use_text_mstff: bool = False,
    ) -> None:
        super().__init__()

        self.context_length = int(context_length)
        self.vocab_size = int(vocab_size)
        self.transformer_width = int(transformer_width)
        self.embed_dim = int(embed_dim)
        self.use_text_mstff = bool(use_text_mstff)

        self.transformer = Transformer(
            width=self.transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.transformer_width,
        )
        self.positional_embedding = nn.Parameter(
            torch.empty(
                self.context_length,
                self.transformer_width,
            )
        )
        self.ln_final = LayerNorm(self.transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(
                self.transformer_width,
                self.embed_dim,
            )
        )
        if self.use_text_mstff:
            self.text_mstff = TextMSTFF(embed_dim=self.transformer_width)
        else:
            self.text_mstff = None

    def build_attention_mask(self) -> torch.Tensor:
        """Build the standard causal CLIP text-attention mask."""
        mask = torch.empty(
            self.context_length,
            self.context_length,
        )
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @staticmethod
    def _validate_prompt_inputs(
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        context_length: int,
    ) -> None:
        if prompts.ndim != 3:
            raise ValueError(
                "prompts must have shape [N, L, D], "
                f"got {tuple(prompts.shape)}."
            )
        if tokenized_prompts.ndim != 2:
            raise ValueError(
                "tokenized_prompts must have shape [N, L], "
                f"got {tuple(tokenized_prompts.shape)}."
            )
        if prompts.shape[:2] != tokenized_prompts.shape:
            raise ValueError(
                "Prompt and token-id shapes do not match: "
                f"prompts={tuple(prompts.shape)}, "
                f"tokenized={tuple(tokenized_prompts.shape)}."
            )
        if prompts.shape[1] != context_length:
            raise ValueError(
                f"Expected context length {context_length}, "
                f"got {prompts.shape[1]}."
            )

    def encode_tokens(
        self,
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        maple_prompts: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all text tokens through ``ln_final``.

        This is the interface used by the learnable text branch before
        Text_MSTFF.

        Returns:
            token_features:
                All token features after ``ln_final``, shape ``[N, L, D]``.
            eot_feature:
                Feature at the EOT position, shape ``[N, D]``.

        Both outputs are before ``text_projection``.
        """
        self._validate_prompt_inputs(
            prompts,
            tokenized_prompts,
            self.context_length,
        )

        positional_embedding = self.positional_embedding.to(
            device=prompts.device,
            dtype=prompts.dtype,
        )
        x = prompts + positional_embedding.unsqueeze(0)

        # [N, L, D] -> [L, N, D]
        x = x.permute(1, 0, 2)

        x = self.transformer(
            x,
            maple_prompts=maple_prompts,
        )

        # [L, N, D] -> [N, L, D]
        x = x.permute(1, 0, 2)
        token_features = self.ln_final(x)

        tokenized_prompts = tokenized_prompts.to(
            device=token_features.device,
            dtype=torch.long,
        )

        # CLIP's EOT token has the largest token ID in each sequence.
        eot_indices = tokenized_prompts.argmax(dim=-1)
        batch_indices = torch.arange(
            token_features.shape[0],
            device=token_features.device,
        )
        eot_feature = token_features[
            batch_indices,
            eot_indices,
        ]

        return token_features, eot_feature

    def encode_with_mstff(
        self,
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        maple_prompts: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode text through ``ln_final``, then Text_MSTFF, then projection.

        Text_MSTFF is applied after the CLIP Text Transformer and ``ln_final``
        but before ``text_projection``, exactly as required.
        """
        token_features, eot_feature = self.encode_tokens(
            prompts,
            tokenized_prompts,
            maple_prompts=maple_prompts,
        )
        if self.text_mstff is None:
            raise RuntimeError(
                "Text_MSTFF is not enabled; pass use_text_mstff=True."
            )
        token_mask = tokenized_prompts.to(
            device=token_features.device,
        ).ne(0)
        fused, _ = self.text_mstff(
            token_features,
            eot_feature,
            token_mask,
        )
        return fused @ self.text_projection

    def forward(
        self,
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        maple_prompts: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Ordinary CLIP text path used by the fixed-template branch."""
        _, eot_feature = self.encode_tokens(
            prompts,
            tokenized_prompts,
            maple_prompts=maple_prompts,
        )
        return eot_feature @ self.text_projection


class TextPromptLearner(nn.Module):
    """Learnable context prompt while preserving class name and EOT tokens."""

    _VALID_CLASS_POSITIONS = {"end", "middle", "front"}

    def __init__(
        self,
        classnames: Sequence[str],
        text_model: CLIPTextEncoder,
        num_prompts: int,
        prompts_init: str = "",
        CSC: bool = False,
        ctx_pos: str = "end",
    ) -> None:
        super().__init__()

        if not classnames:
            raise ValueError("classnames cannot be empty.")
        if ctx_pos not in self._VALID_CLASS_POSITIONS:
            raise ValueError(
                f"ctx_pos must be one of {sorted(self._VALID_CLASS_POSITIONS)}, "
                f"got {ctx_pos!r}."
            )

        tokenizer = _Tokenizer()
        classnames = [
            str(name).replace("_", " ").strip()
            for name in classnames
        ]

        if any(not name for name in classnames):
            raise ValueError("classnames contains an empty class name.")

        num_classes = len(classnames)
        num_context_tokens = int(num_prompts)
        context_dimension = int(
            text_model.ln_final.weight.shape[0]
        )

        context_initialization = prompts_init.strip()

        if context_initialization:
            context_initialization = (
                context_initialization.replace("_", " ")
            )
            num_context_tokens = len(
                context_initialization.split()
            )

            initialized_prompt = tokenize(
                context_initialization,
                context_length=text_model.context_length,
            )

            with torch.no_grad():
                initialized_embedding = text_model.token_embedding(
                    initialized_prompt
                )

            context_vectors = initialized_embedding[
                0,
                1 : 1 + num_context_tokens,
                :,
            ].clone()
            prompt_prefix = context_initialization
        else:
            if num_context_tokens <= 0:
                raise ValueError(
                    "num_prompts must be positive when prompts_init is empty."
                )

            if CSC:
                print("Initializing class-specific contexts")
                context_vectors = torch.empty(
                    num_classes,
                    num_context_tokens,
                    context_dimension,
                )
            else:
                print("Initializing a generic context")
                context_vectors = torch.empty(
                    num_context_tokens,
                    context_dimension,
                )

            nn.init.normal_(context_vectors, std=0.02)
            prompt_prefix = " ".join(
                ["X"] * num_context_tokens
            )

        print(f'Initial context: "{prompt_prefix}"')
        print(
            "Number of context words (tokens): "
            f"{num_context_tokens}"
        )

        # This is the only trainable part of the Prompt Learner.
        self.ctx = nn.Parameter(context_vectors)

        class_name_token_lengths = [
            len(tokenizer.encode(name))
            for name in classnames
        ]

        # The period helps CLIP receive a natural complete prompt.
        prompt_texts = [
            f"{prompt_prefix} {class_name}."
            for class_name in classnames
        ]
        tokenized_prompts = tokenize(
            prompt_texts,
            context_length=text_model.context_length,
        )

        with torch.no_grad():
            embedding = text_model.token_embedding(
                tokenized_prompts
            )

        # SOS is fixed. ``token_suffix`` contains the class-name tokens,
        # EOT and padding tokens, so class names and EOT are never removed.
        self.register_buffer(
            "token_prefix",
            embedding[:, :1, :],
        )
        self.register_buffer(
            "token_suffix",
            embedding[:, 1 + num_context_tokens :, :],
        )
        self.register_buffer(
            "tokenized_prompts",
            tokenized_prompts,
            persistent=False,
        )

        self.n_cls = num_classes
        self.n_ctx = num_context_tokens
        self.name_lens = class_name_token_lengths
        self.class_token_position = ctx_pos

    def forward(self) -> torch.Tensor:
        context = self.ctx

        if context.ndim == 2:
            context = context.unsqueeze(0).expand(
                self.n_cls,
                -1,
                -1,
            )

        prefix = self.token_prefix.to(
            device=context.device,
            dtype=context.dtype,
        )
        suffix = self.token_suffix.to(
            device=context.device,
            dtype=context.dtype,
        )

        if self.class_token_position == "end":
            return torch.cat(
                [prefix, context, suffix],
                dim=1,
            )

        prompts: List[torch.Tensor] = []

        if self.class_token_position == "middle":
            first_context_length = self.n_ctx // 2

            for class_index in range(self.n_cls):
                class_name_length = self.name_lens[class_index]

                prefix_i = prefix[
                    class_index : class_index + 1
                ]
                class_i = suffix[
                    class_index : class_index + 1,
                    :class_name_length,
                ]
                suffix_i = suffix[
                    class_index : class_index + 1,
                    class_name_length:,
                ]
                context_first = context[
                    class_index : class_index + 1,
                    :first_context_length,
                ]
                context_second = context[
                    class_index : class_index + 1,
                    first_context_length:,
                ]

                prompts.append(
                    torch.cat(
                        [
                            prefix_i,
                            context_first,
                            class_i,
                            context_second,
                            suffix_i,
                        ],
                        dim=1,
                    )
                )

        elif self.class_token_position == "front":
            for class_index in range(self.n_cls):
                class_name_length = self.name_lens[class_index]

                prefix_i = prefix[
                    class_index : class_index + 1
                ]
                class_i = suffix[
                    class_index : class_index + 1,
                    :class_name_length,
                ]
                suffix_i = suffix[
                    class_index : class_index + 1,
                    class_name_length:,
                ]
                context_i = context[
                    class_index : class_index + 1
                ]

                prompts.append(
                    torch.cat(
                        [
                            prefix_i,
                            class_i,
                            context_i,
                            suffix_i,
                        ],
                        dim=1,
                    )
                )

        else:
            raise RuntimeError(
                "Unexpected class-token position: "
                f"{self.class_token_position!r}."
            )

        return torch.cat(prompts, dim=0)


def _get_clones(module: nn.Module, number: int) -> nn.ModuleList:
    return nn.ModuleList(
        [copy.deepcopy(module) for _ in range(number)]
    )
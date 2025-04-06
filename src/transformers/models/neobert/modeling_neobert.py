from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import scaled_dot_product_attention

from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_xformers_available,
    logging,
)
from .configuration_neobert import NeoBertConfig


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
else:
    FLASH_ATTN_AVAILABLE = False

if is_xformers_available():
    from xformers.ops import SwiGLU
else:
    raise ImportError("Xformers is not available. Please install it with `pip install xformers==0.0.28.post3`.")

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "chandar-lab/NeoBERT"
_CONFIG_FOR_DOC = "NeoBertConfig"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.
    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    assert freqs_cis.shape[1:] == (x.shape[1], x.shape[-1])
    return freqs_cis.contiguous().unsqueeze(2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.
    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class EncoderBlock(nn.Module):
    """NeoBERT encoder block."""

    def __init__(self, config: NeoBertConfig):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

        # Feedforward network
        multiple_of = 8
        intermediate_size = int(2 * config.intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)

        # Layer norms
        self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        output_attentions: bool,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
    ):
        # Attention
        attn_output, attn_weights = self._att_block(
            self.attention_norm(x), attention_mask, freqs_cis, output_attentions, max_seqlen, cu_seqlens
        )

        # Residual
        x = x + attn_output

        # Feed-forward
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        output_attentions: bool,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = (
            self.qkv(x)
            .view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3)
            .chunk(3, axis=-1)
        )

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Attn block
        attn_weights = None

        # Flash attention if the tensors are packed
        if cu_seqlens is not None:
            attn = flash_attn_varlen_func(
                q=xq.squeeze(0),
                k=xk.squeeze(0),
                v=xv.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
            )
        # Eager attention if attention weights are needed in the output
        elif output_attentions:
            attn_weights = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            if attention_mask is not None:
                attn_weights = attn_weights * attention_mask
            attn_weights = attn_weights.softmax(-1)
            attn = attn_weights @ xv.permute(0, 2, 1, 3)
            attn = attn.transpose(1, 2)
        # Fall back to SDPA otherwise
        else:
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attention_mask.bool(),
                dropout_p=0,
            ).transpose(1, 2)

        return self.wo(
            attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.config.dim_head)
        ), attn_weights


NEOBERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`NeoBertConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare NeoBERT Model outputting raw hidden-states without any specific head on top.",
    NEOBERT_START_DOCSTRING,
)
class NeoBertPreTrainedModel(PreTrainedModel):
    config_class = NeoBertConfig
    base_model_prefix = "model"
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)


NEOBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. With Flash Attention 2.0, padding will be ignored
            by default should you provide it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The NeoBERT Model outputting raw hidden-states without any specific head on top.",
    NEOBERT_START_DOCSTRING,
)
class NeoBertModel(NeoBertPreTrainedModel):
    config_class = NeoBertConfig

    def __init__(self, config: NeoBertConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # Ensures freqs_cis is moved to the same devices as the model. Non-persistent buffers are not saved in the state_dict.
        freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        self.layer_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(NEOBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # Initialize
        hidden_states, attentions = [], []

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if attention_mask is not None:
            attention_mask = (
                attention_mask.unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, self.config.num_attention_heads, attention_mask.size(-1), 1)
            )

        # Checks to be done if inputs are packed sequences
        if cu_seqlens is not None:
            assert (
                FLASH_ATTN_AVAILABLE
            ), "Flash-attention is not available. Please ''pip install flash_attn'', or provide un-packed sequences."
            assert not output_attentions, "Output attentions is not supported when sequences are packed."
            assert max_seqlen is not None, "Missing max_seqlen. It must be provided when cu_seqlens are not None."
            assert (input_ids if input_ids is not None else inputs_embeds).shape[
                0
            ] == 1, "Cumulative sequence lengths are provided but inputs are not packed."
            assert (
                input_ids if input_ids is not None else inputs_embeds
            ).is_cuda, "Packing uses an implementation of flash-attention and is only supported on GPU."

        # RoPE
        freqs_cis = (
            self.freqs_cis[position_ids]
            if position_ids is not None
            else self.freqs_cis[: (input_ids if input_ids is not None else inputs_embeds).shape[1]].unsqueeze(0)
        )

        # Embedding
        x = self.encoder(input_ids) if input_ids is not None else inputs_embeds

        # Transformer encoder
        for layer in self.transformer_encoder:
            x, attn = layer(x, attention_mask, freqs_cis, output_attentions, max_seqlen, cu_seqlens)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Final normalization layer
        x = self.layer_norm(x)

        # Return the output of the last hidden layer
        if not return_dict:
            return tuple(v for v in [x, hidden_states, attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
        )


@add_start_docstrings(
    "The NeoBERT Model with a decoder head on top that is used for masked language modeling.",
    NEOBERT_START_DOCSTRING,
)
class NeoBertForMaskedLM(NeoBertPreTrainedModel):
    config_class = NeoBertConfig

    def __init__(self, config: NeoBertConfig):
        super().__init__(config)

        self.config = config

        self.model = NeoBertModel(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    @add_start_docstrings_to_model_forward(NEOBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.decoder(output.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        if not return_dict:
            outputs = (logits,)
            return ((loss,) + outputs) if loss is not None else outputs

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions if output_attentions else None,
        )


@add_start_docstrings(
    "The NeoBERT Model with a sequence classification head on top that performs pooling.",
    NEOBERT_START_DOCSTRING,
)
class NeoBertForSequenceClassification(NeoBertPreTrainedModel):
    config_class = NeoBertConfig

    def __init__(self, config: NeoBertConfig):
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBertModel(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    @add_start_docstrings_to_model_forward(NEOBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = output.last_hidden_state

        x = hidden_states[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions if output_attentions else None,
        )


__all__ = [
    "NeoBertModel",
    "NeoBertPreTrainedModel",
    "NeoBertForMaskedLM",
    "NeoBertForSequenceClassification",
]

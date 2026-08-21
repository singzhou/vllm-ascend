from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from vllm_ascend.patch.worker.patch_draft_quarot import get_rotataion_matrix, get_rotation_path


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    if rotation_weight.dtype != torch.float32:
        rotation_weight = rotation_weight.to(torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32)
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    return processed_weight.to(ori_dtype)


class DSparkConfidenceHead(nn.Module):
    def __init__(self, config, prefix: str) -> None:
        super().__init__()

        rank = int(getattr(config, "markov_rank", getattr(config, "dspark_markov_rank", 256)))
        self.proj = ReplicatedLinear(
            config.hidden_size + rank,
            1,
            bias=True,  # released dspark_qwen3_*_block7 ckpt has confidence_head.proj.bias
            params_dtype=torch.float32,
            quant_config=None,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hidden_states, markov_embeds], dim=-1)
        confidence, _ = self.proj(x.float())  # ReplicatedLinear returns (output, bias)
        return confidence.squeeze(-1)


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    #: Whether the checkpoint actually carried a ``d2t`` vocabulary mapping.
    #:
    #: A draft declaring ``draft_vocab_size`` allocates ``draft_id_to_target_id``
    #: zero-filled, and the loader skips the parameter when the checkpoint has no
    #: ``d2t``. Its contents therefore cannot tell "nothing was loaded" apart from
    #: the legitimate mapping that keeps target ids ``0..K-1``, since both are all
    #: zeros -- training resolves the same ambiguity with ``t2d``, which serving
    #: drops. The weight stream can tell them apart, and this is the last place
    #: that sees it.
    has_draft_id_mapping: bool = False

    #: Whether the checkpoint carried ``lm_head`` weights of its own.
    #:
    #: Kept separately from ``has_own_lm_head`` because the two answer different
    #: questions once the vocabulary is reduced: this one is "were the weights
    #: loaded", while ``has_own_lm_head`` decides "may the target head replace
    #: it" -- and for a reduced vocabulary the answer to the second is always no.
    has_own_lm_head_weights: bool = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        self.enable_confidence_head = bool(
            getattr(config, "enable_confidence_head", hasattr(config, "markov_head_type"))
        )
        if self.enable_confidence_head:
            model_prefix = maybe_prefix(prefix, "model")
            self.model.confidence_head = DSparkConfidenceHead(
                config=config,
                prefix=maybe_prefix(model_prefix, "confidence_head"),
            )
        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None

    def _note_checkpoint_contents(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Record which optional keys the stream carries, passing it through."""
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.has_draft_id_mapping = True
            if "lm_head" in name:
                self.has_own_lm_head_weights = True
            yield name, loaded_weight

    @staticmethod
    def _get_confidence_relative_name(
        checkpoint_name: str,
    ) -> str | None:
        marker = "confidence_head."
        marker_pos = checkpoint_name.find(marker)

        if marker_pos == -1:
            return None

        return checkpoint_name[marker_pos + len(marker) :]

    def confidence_logits(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")

        return self.model.confidence_head(
            hidden_states,
            markov_embeds,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        all_weights = list(weights)
        if self.rotation_path is not None:
            processed_weights: list[tuple[str, torch.Tensor]] = []
            rotation_weight = get_rotataion_matrix(self.rotation_path)
            for name, loaded_weight in all_weights:
                if "fc." in name:
                    loaded_weight = process_weight(loaded_weight, rotation_weight)
                processed_weights.append((name, loaded_weight))
            all_weights = processed_weights

        base_weights: list[tuple[str, torch.Tensor]] = []
        confidence_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in all_weights:
            confidence_name = self._get_confidence_relative_name(name)
            if confidence_name is None:
                base_weights.append((name, loaded_weight))
            else:
                confidence_weights.append((confidence_name, loaded_weight))

        super().load_weights(self._note_checkpoint_contents(base_weights))

        if self.draft_id_to_target_id is not None:
            # ``has_own_lm_head`` drives exactly one decision in the draft-model
            # loaders: may the target's LM head replace this model's. For a
            # reduced vocabulary the answer is never yes -- the target head spans
            # the full vocabulary, and the logits processor would then quietly
            # slice its first draft_vocab_size columns instead of the ones the
            # mapping keeps, proposing plausible but wrong tokens. The head this
            # model built is the right shape; the speculator fills it from the
            # kept target rows. ``has_own_lm_head_weights`` remains the record of
            # what the checkpoint actually shipped.
            self.has_own_lm_head = True

        if not self.enable_confidence_head:
            return

        if not confidence_weights:
            raise RuntimeError(
                "The DSpark confidence head is enabled, but no confidence_head tensors were found in the checkpoint."
            )

        confidence_weights.sort(key=lambda item: item[0])
        loaded_parameters = AutoWeightsLoader(self.model.confidence_head).load_weights(confidence_weights)
        expected_parameters = set(self.model.confidence_head.state_dict().keys())
        missing_parameters = expected_parameters - loaded_parameters

        if missing_parameters:
            raise RuntimeError(
                "Failed to load all confidence-head "
                "parameters. Missing: "
                f"{sorted(missing_parameters)}; loaded: "
                f"{sorted(loaded_parameters)}"
            )

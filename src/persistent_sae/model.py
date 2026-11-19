"""Persistent sparse autoencoder.

For a sequence of hidden states x_1, ..., x_T the model computes

    a_t = BatchTopK(ReLU(W_enc (x_t - b_dec) + b_enc))      # sparse evidence
    h_t = lambda * h_{t-1} + (1 - lambda) * a_t               # persistent state
    x_hat_t = W_dec h_t + b_dec                               # reconstruction

where lambda in (0, max_persistence) is a learned persistence coefficient per
feature (or per group of features).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch as t
import torch.nn as nn
import torch.nn.functional as F


@t.no_grad()
def set_decoder_norm_to_unit_norm(weight: t.Tensor, activation_dim: int, dict_size: int) -> t.Tensor:
    """Normalise the columns of an ``nn.Linear`` decoder weight of shape (activation_dim, dict_size)."""
    assert weight.shape == (activation_dim, dict_size), weight.shape
    eps = t.finfo(weight.dtype).eps
    weight.data /= t.norm(weight.data, dim=0, keepdim=True) + eps
    return weight.data


# Buffers written by the research codebase for options that are not part of the
# released model. A legacy checkpoint is accepted only if each one holds the
# value that corresponds to the plain EMA Persistent SAE.
_LEGACY_REQUIRED_VALUES = {
    "ema_update": True,
    "separated_paths": False,
    "learn_memory_strength": False,
    "learn_memory_gate": False,
    "state_k": -1,
}
_LEGACY_DROPPED_KEYS = ("state_threshold", "persistence_mode_code", "persistence_group_index")
_LEGACY_UNSUPPORTED_PARAMS = ("memory_strength_logit", "memory_gate_bias")


def convert_legacy_state_dict(state_dict: dict[str, t.Tensor]) -> dict[str, t.Tensor]:
    """Convert a research-code ``PersistentDynamicBatchTopKSAE`` state dict to ``PersistentSAE``."""
    state_dict = dict(state_dict)
    # Research checkpoints without this flag used a non-EMA update and have otherwise identical keys.
    if "ema_update" not in state_dict:
        raise ValueError("Checkpoint has no ema_update flag; only EMA Persistent SAEs are supported.")
    for key, expected in _LEGACY_REQUIRED_VALUES.items():
        if key in state_dict:
            value = state_dict[key].item() if key == "ema_update" else state_dict.pop(key).item()
            if value != expected:
                raise ValueError(f"Checkpoint uses unsupported option {key}={value!r} (expected {expected!r}).")
    for key in _LEGACY_UNSUPPORTED_PARAMS:
        if key in state_dict:
            raise ValueError(f"Checkpoint contains unsupported parameter {key!r}.")
    for key in _LEGACY_DROPPED_KEYS:
        state_dict.pop(key, None)
    return state_dict


class PersistentSAE(nn.Module):
    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        init_persistence: float = 0.5,
        max_persistence: float = 0.999,
        num_persistence_groups: int | None = None,
        persistence_init: str = "constant",
    ):
        """
        Args:
            activation_dim: width of the hidden states being reconstructed.
            dict_size: number of features.
            k: average number of active evidence features per token (BatchTopK).
            init_persistence: initial persistence for ``persistence_init="constant"``.
            max_persistence: upper bound on every persistence coefficient.
            num_persistence_groups: learn one coefficient per contiguous group of
                features instead of one per feature. ``None`` means featurewise.
            persistence_init: initialisation of the persistence logits, one of
                ``constant``, ``kaiming_uniform`` or ``xavier_uniform``.
        """
        super().__init__()
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)

        if not (isinstance(k, int) and k > 0):
            raise ValueError(f"k={k} must be a positive integer.")
        if not (0.0 < init_persistence < max_persistence < 1.0):
            raise ValueError("Expected 0 < init_persistence < max_persistence < 1.")
        if persistence_init not in {"constant", "kaiming_uniform", "xavier_uniform"}:
            raise ValueError("persistence_init must be 'constant', 'kaiming_uniform', or 'xavier_uniform'.")
        num_coefficients = self.dict_size if num_persistence_groups is None else int(num_persistence_groups)
        if not 1 <= num_coefficients <= self.dict_size:
            raise ValueError(f"num_persistence_groups must be in [1, {self.dict_size}].")

        self.register_buffer("k", t.tensor(k, dtype=t.int))
        # BatchTopK is replaced by a fixed activation threshold at inference;
        # -1 means the threshold has not been calibrated yet.
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))
        self.register_buffer("max_persistence", t.tensor(float(max_persistence), dtype=t.float32))
        # Always True; distinguishes these checkpoints from early non-EMA research checkpoints.
        self.register_buffer("ema_update", t.tensor(True, dtype=t.bool))
        group_index = t.div(
            t.arange(self.dict_size, dtype=t.long) * num_coefficients,
            self.dict_size,
            rounding_mode="floor",
        )
        self.register_buffer("persistence_group_index", group_index, persistent=False)

        self.decoder = nn.Linear(self.dict_size, self.activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight.data, self.activation_dim, self.dict_size
        )
        self.encoder = nn.Linear(self.activation_dim, self.dict_size)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(t.zeros(self.activation_dim))

        persistence_logit = t.empty((num_coefficients, 1), dtype=t.float32)
        if persistence_init == "constant":
            init_ratio = min(max(init_persistence / max_persistence, 1e-4), 1.0 - 1e-4)
            persistence_logit.fill_(math.log(init_ratio / (1.0 - init_ratio)))
        elif persistence_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(persistence_logit, a=math.sqrt(5))
        else:
            nn.init.xavier_uniform_(persistence_logit)
        self.persistence_logit = nn.Parameter(persistence_logit.squeeze(1))

    @property
    def num_persistence_groups(self) -> int:
        return int(self.persistence_logit.numel())

    def persistence(self) -> t.Tensor:
        """Per-feature persistence coefficients lambda, shape (dict_size,)."""
        coefficients = t.sigmoid(self.persistence_logit) * self.max_persistence
        return coefficients.index_select(0, self.persistence_group_index)

    def encode_evidence(
        self,
        x: t.Tensor,
        token_mask: t.Tensor | None = None,
        use_threshold: bool = True,
        return_post_relu: bool = False,
    ):
        """Sparse evidence a_t for every token, shape (..., dict_size).

        With ``use_threshold=False`` the BatchTopK selection is applied across all
        unmasked tokens in ``x`` (training); otherwise the calibrated threshold is used.
        """
        prefix_shape = tuple(x.shape[:-1])
        flat_x = x.reshape(-1, x.size(-1))
        post_relu = F.relu(self.encoder(flat_x - self.b_dec))
        if token_mask is None:
            flat_mask = t.ones(flat_x.size(0), dtype=t.bool, device=flat_x.device)
        else:
            flat_mask = token_mask.reshape(-1).to(dtype=t.bool, device=flat_x.device)

        if use_threshold:
            encoded = post_relu * (post_relu > self.threshold)
            encoded = encoded * flat_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        else:
            encoded = t.zeros_like(post_relu)
            valid_post_relu = post_relu[flat_mask]
            if valid_post_relu.numel() > 0:
                flat_post_relu = valid_post_relu.flatten()
                topk = flat_post_relu.topk(int(self.k.item()) * valid_post_relu.size(0), sorted=False, dim=-1)
                valid_encoded = (
                    t.zeros_like(flat_post_relu).scatter_(-1, topk.indices, topk.values).reshape_as(valid_post_relu)
                )
                encoded[flat_mask] = valid_encoded

        encoded = encoded.reshape(*prefix_shape, self.dict_size)
        if return_post_relu:
            return encoded, post_relu.reshape(*prefix_shape, self.dict_size)
        return encoded

    def accumulate_state(self, evidence: t.Tensor, token_mask: t.Tensor | None = None) -> t.Tensor:
        """Run the recurrence h_t = lambda h_{t-1} + (1 - lambda) a_t over (batch, seq, dict_size)."""
        orig_dtype = evidence.dtype
        # Upcast to fp32 only when grad is on (training): the scan is
        # well-conditioned in practice, but bf16 accumulation across many
        # additions can drift enough to hurt the gradient.  At inference
        # fp32 just doubles memory for no gain.
        scan_dtype = t.float32 if t.is_grad_enabled() else evidence.dtype
        persistence = self.persistence().to(dtype=scan_dtype)
        seq_len = evidence.size(1)

        # Hillis-Steele parallel scan for h_t = u_t + p * h_{t-1} with u_t = (1-p) * a_t.
        # The coefficient on h_{t-step} at iteration `step` is exactly p**step,
        # uniform across batch and position — no (B, L, D) coeff tensor needed.
        #
        # We rebuild `state` out-of-place via cat each iteration rather than
        # in-place slice-assigning.  In-place slice writes bump the autograd
        # version counter on `state`, and earlier slices of `state` saved for
        # backward (e.g. the lagged read `state[:, :-step, :]`) then fail the
        # version check.  The cat form avoids in-place mutation entirely.
        state = evidence.to(dtype=scan_dtype) * (1 - persistence)
        p_pow = persistence  # p^1
        step = 1
        while step < seq_len:
            tail = state[:, step:, :] + p_pow * state[:, :-step, :]
            state = t.cat([state[:, :step, :], tail], dim=1)
            step *= 2
            if step < seq_len:
                p_pow = p_pow * p_pow  # p^(2*prev_step) = p^step

        if token_mask is not None:
            state = state * token_mask.unsqueeze(-1).to(dtype=state.dtype)
        return state.to(dtype=orig_dtype)

    def carried_state(self, state: t.Tensor) -> t.Tensor:
        """The part of h_t carried over from earlier tokens, lambda * h_{t-1}."""
        previous = t.cat([t.zeros_like(state[:, :1, :]), state[:, :-1, :]], dim=1)
        return previous * self.persistence().to(dtype=state.dtype)

    def encode_state(
        self,
        x: t.Tensor,
        token_mask: t.Tensor | None = None,
        use_threshold: bool = True,
        use_state_threshold: bool = False,
    ) -> t.Tensor:
        """Persistent state h_t for (seq, d) or (batch, seq, d) hidden states.

        With ``use_state_threshold`` the state is pruned with the evidence threshold,
        giving a sparser readout for analysis.
        """
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)
            token_mask = token_mask.unsqueeze(0) if token_mask is not None else None
        evidence = self.encode_evidence(x, token_mask=token_mask, use_threshold=use_threshold)
        state = self.accumulate_state(evidence, token_mask=token_mask)
        if use_state_threshold and self.threshold >= 0:
            state = state * (state > self.threshold.to(dtype=state.dtype))
        return state.squeeze(0) if unbatched else state

    def decode(self, state: t.Tensor) -> t.Tensor:
        return self.decoder(state) + self.b_dec

    def forward(self, x: t.Tensor, token_mask: t.Tensor | None = None, output_features: bool = False):
        state = self.encode_state(x, token_mask=token_mask)
        x_hat = self.decode(state)
        if output_features:
            return x_hat, state
        return x_hat

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | None = None) -> "PersistentSAE":
        """Load an ``ae.pt`` state dict, including checkpoints from the research codebase."""
        state_dict = convert_legacy_state_dict(t.load(path, map_location="cpu", weights_only=True))
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        num_coefficients = int(state_dict["persistence_logit"].numel())
        autoencoder = cls(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=int(state_dict["k"].item()),
            max_persistence=float(state_dict["max_persistence"].item()),
            num_persistence_groups=None if num_coefficients == dict_size else num_coefficients,
        )
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

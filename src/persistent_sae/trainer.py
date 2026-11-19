"""Training step for Persistent SAEs: BatchTopK evidence, auxiliary dead-feature loss, threshold calibration."""

from __future__ import annotations

import torch as t

from .model import PersistentSAE, set_decoder_norm_to_unit_norm


@t.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    weight: t.Tensor, grad: t.Tensor, activation_dim: int, dict_size: int
) -> t.Tensor:
    """Project out the gradient component that would change decoder column norms."""
    assert weight.shape == (activation_dim, dict_size), weight.shape
    normed = weight / (t.norm(weight, dim=0, keepdim=True) + 1e-6)
    parallel = (grad * normed).sum(dim=0)
    grad -= parallel.unsqueeze(0) * normed
    return grad


def get_lr_schedule(total_steps: int, warmup_steps: int, decay_start: int | None = None):
    """Linear warm-up, constant, then optional linear decay to zero."""
    assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."
    if decay_start is not None:
        assert warmup_steps < decay_start < total_steps, "decay_start must be in (warmup_steps, steps)."

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if decay_start is not None and step >= decay_start:
            return (total_steps - step) / (total_steps - decay_start)
        return 1.0

    return lr_schedule


class PersistentSAETrainer:
    def __init__(
        self,
        steps: int,
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        lr: float = 3e-4,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: int | None = None,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        init_persistence: float = 0.5,
        max_persistence: float = 0.999,
        num_persistence_groups: int | None = None,
        persistence_init: str = "constant",
        seed: int | None = None,
        device: str | None = None,
    ):
        self.steps = steps
        self.layer = layer
        self.lm_name = lm_name
        self.lr = lr
        self.auxk_alpha = auxk_alpha
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.init_persistence = float(init_persistence)
        self.persistence_init = persistence_init
        self.seed = seed

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = PersistentSAE(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            init_persistence=init_persistence,
            max_persistence=max_persistence,
            num_persistence_groups=num_persistence_groups,
            persistence_init=persistence_init,
        )
        self.device = device or ("cuda" if t.cuda.is_available() else "cpu")
        self.ae.to(self.device)

        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=self.device)
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1.0

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def get_auxiliary_loss(self, residual_btd: t.Tensor, post_relu_btf: t.Tensor, token_mask: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())
        if self.dead_features <= 0:
            self.pre_norm_auxk_loss = -1.0
            return t.tensor(0, dtype=residual_btd.dtype, device=residual_btd.device)

        flat_post_relu = post_relu_btf.reshape(-1, post_relu_btf.size(-1))
        flat_residual = residual_btd.reshape(-1, residual_btd.size(-1))
        flat_mask = token_mask.reshape(-1).to(dtype=t.bool, device=flat_post_relu.device)

        k_aux = min(self.top_k_aux, self.dead_features)
        valid_post_relu = flat_post_relu[flat_mask]
        valid_residual = flat_residual[flat_mask]
        auxk_latents = t.where(dead_features[None], valid_post_relu, -t.inf)
        auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
        auxk_valid_evidence = t.zeros_like(valid_post_relu).scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
        auxk_evidence = t.zeros_like(flat_post_relu)
        auxk_evidence[flat_mask] = auxk_valid_evidence
        auxk_evidence = auxk_evidence.reshape_as(post_relu_btf)

        auxk_state = self.ae.accumulate_state(auxk_evidence, token_mask=token_mask)
        # Use decoder without bias: the residual already has b_dec subtracted, so adding it again
        # would shift the aux target by 2*b_dec instead of b_dec.
        x_reconstruct_aux = self.ae.decoder(auxk_state.reshape(-1, self.ae.dict_size))[flat_mask]
        l2_loss_aux = (valid_residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
        self.pre_norm_auxk_loss = float(l2_loss_aux.item())

        residual_mu = valid_residual.mean(dim=0, keepdim=True)
        loss_denom = (valid_residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
        # Clamp denominator away from zero: loss_denom is normally O(d_model),
        # so 1.0 is a safe floor that only activates when residual variance is degenerate.
        return (l2_loss_aux / loss_denom.clamp_min(1.0)).nan_to_num(0.0)

    @t.no_grad()
    def update_threshold(self, evidence: t.Tensor):
        device_type = "cuda" if evidence.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False):
            active = evidence[evidence > 0]
            min_activation = active.min().to(dtype=t.float32) if active.numel() else 0.0
            if self.ae.threshold < 0:
                self.ae.threshold.fill_(min_activation)
            else:
                self.ae.threshold.mul_(self.threshold_beta).add_((1 - self.threshold_beta) * min_activation)

    def loss(self, x: t.Tensor, token_mask: t.Tensor, step: int | None = None, return_stats: bool = False):
        """Reconstruction + auxiliary loss on a (batch, seq, d) block of padded sequences."""
        evidence, post_relu = self.ae.encode_evidence(
            x, token_mask=token_mask, use_threshold=False, return_post_relu=True
        )
        flat_mask = token_mask.reshape(-1)
        valid_evidence = evidence.reshape(-1, self.ae.dict_size)[flat_mask]
        if step is not None and step > self.threshold_start_step:
            self.update_threshold(valid_evidence)

        state = self.ae.accumulate_state(evidence, token_mask=token_mask)
        x_hat = self.ae.decode(state)
        residual = x - x_hat
        valid_state = state.reshape(-1, self.ae.dict_size)[flat_mask]
        valid_residual = residual.reshape(-1, residual.size(-1))[flat_mask]

        did_fire = (valid_evidence.sum(dim=0) > 0) | (valid_state.abs().sum(dim=0) > 0)
        self.num_tokens_since_fired += int(flat_mask.sum().item())
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = valid_residual.float().pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(residual.detach(), post_relu, token_mask)
        loss = l2_loss + self.auxk_alpha * auxk_loss
        if not return_stats:
            return loss

        with t.no_grad():
            valid_x = x.reshape(-1, x.size(-1))[flat_mask].float()
            total_variance = valid_x.var(dim=0).sum()
            residual_variance = valid_residual.float().var(dim=0).sum()
            # Carried-mass fraction: share of the state's magnitude contributed by lambda * h_{t-1}.
            carried = self.ae.carried_state(state.float()).reshape(-1, self.ae.dict_size)[flat_mask]
            persistence = self.ae.persistence().float()
            stats = {
                "loss": float(loss.item()),
                "l2_loss": float(l2_loss.item()),
                "auxk_loss": float(auxk_loss.item()),
                "pre_norm_auxk_loss": self.pre_norm_auxk_loss,
                "frac_variance_explained": float((1 - residual_variance / total_variance.clamp_min(1e-8)).item()),
                "evidence_l0": float((valid_evidence != 0).float().sum(dim=-1).mean().item()),
                "state_l0": float((valid_state != 0).float().sum(dim=-1).mean().item()),
                "carried_mass_fraction": float(
                    (carried.sum() / valid_state.float().sum().clamp_min(1e-8)).item()
                ),
                "dead_features": self.dead_features,
                "threshold": float(self.ae.threshold.item()),
                "persistence_mean": float(persistence.mean().item()),
                "persistence_q05": float(t.quantile(persistence, 0.05).item()),
                "persistence_q50": float(t.quantile(persistence, 0.50).item()),
                "persistence_q95": float(t.quantile(persistence, 0.95).item()),
                "lr": float(self.scheduler.get_last_lr()[0]),
            }
        return loss, stats

    def update(self, step: int, x: t.Tensor, token_mask: t.Tensor, return_stats: bool = False):
        """One optimiser step. Returns the stats dict when ``return_stats`` is set, else None."""
        x = x.to(self.device)
        token_mask = token_mask.to(self.device)
        if step == 0:
            median_points = x.reshape(-1, x.size(-1))[token_mask.reshape(-1)]
            self.ae.b_dec.data = self.geometric_median(median_points).to(self.ae.b_dec.dtype)

        out = self.loss(x, token_mask, step=step, return_stats=return_stats)
        loss, stats = out if return_stats else (out, None)
        if not loss.isfinite():
            self.optimizer.zero_grad()
            return stats
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight, self.ae.decoder.weight.grad, self.ae.activation_dim, self.ae.dict_size
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        set_decoder_norm_to_unit_norm(self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size)
        return stats

    @property
    def config(self) -> dict:
        return {
            "trainer_class": "PersistentSAETrainer",
            "dict_class": "PersistentSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": int(self.ae.k.item()),
            "init_persistence": self.init_persistence,
            "max_persistence": float(self.ae.max_persistence.item()),
            "num_persistence_groups": self.ae.num_persistence_groups,
            "persistence_init": self.persistence_init,
            "layer": self.layer,
            "lm_name": self.lm_name,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1).clamp_min(1e-6)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break
        return guess

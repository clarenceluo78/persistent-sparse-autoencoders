"""Train a Persistent SAE on the residual stream of a Hugging Face causal language model.

Example (paper configuration for Pythia-160M-deduped, layer 8):

    python -m persistent_sae.train --model EleutherAI/pythia-160m-deduped --layer 8 --refresh-batch-size 1024
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .buffer import SequenceActivationBuffer, StreamingTextIterator
from .trainer import PersistentSAETrainer

TRAINING_STATE_FORMAT = 1
# Trainer settings that must match when resuming from a training state.
RESUME_COMPATIBILITY_KEYS = (
    "trainer_class", "activation_dim", "dict_size", "k", "max_persistence",
    "num_persistence_groups", "persistence_init", "layer", "lm_name",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model = parser.add_argument_group("language model and data")
    model.add_argument("--model", required=True, help="Hugging Face model id, e.g. EleutherAI/pythia-160m-deduped.")
    model.add_argument("--layer", type=int, required=True, help="Decoder block whose output residual stream is used.")
    model.add_argument("--dataset", default="monology/pile-uncopyrighted")
    model.add_argument("--dataset-split", default="train")
    model.add_argument("--ctx-len", type=int, default=256, help="Tokens per training sequence (first token dropped).")
    model.add_argument("--seq-batch-size", type=int, default=8, help="Sequences per optimiser step.")
    model.add_argument("--n-ctxs", type=int, default=1000, help="Sequences held in the activation buffer.")
    model.add_argument("--refresh-batch-size", type=int, default=512, help="Documents per LM forward pass.")

    sae = parser.add_argument_group("Persistent SAE")
    sae.add_argument("--dict-size", type=int, default=16384)
    sae.add_argument("--k", type=int, default=20, help="BatchTopK evidence budget per token.")
    sae.add_argument("--init-persistence", type=float, default=0.5)
    sae.add_argument("--max-persistence", type=float, default=0.999)
    sae.add_argument(
        "--num-persistence-groups",
        type=int,
        default=None,
        help="Share persistence coefficients across this many contiguous feature groups (default: featurewise).",
    )
    sae.add_argument(
        "--persistence-init",
        choices=("constant", "kaiming_uniform", "xavier_uniform"),
        default="constant",
        help="Initialisation of the persistence logits; 'constant' uses --init-persistence.",
    )

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--max-steps", type=int, default=244_140)
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--warmup-steps", type=int, default=1000)
    optim.add_argument("--seed", type=int, default=0)

    io = parser.add_argument_group("checkpoints and logging")
    io.add_argument("--save-root", default="checkpoints")
    io.add_argument("--run-name", default=None, help="Defaults to a name built from the model and hyperparameters.")
    io.add_argument("--checkpoint-start-step", type=int, default=1000)
    io.add_argument("--checkpoint-every-steps", type=int, default=None,
                    help="Additionally save a resumable checkpoint every N steps.")
    io.add_argument("--resume", default=None, help="Resume from a training_state*.pt file.")
    io.add_argument("--log-steps", type=int, default=50)
    io.add_argument("--wandb-project", default=None, help="Log to Weights & Biases when set.")
    io.add_argument("--wandb-entity", default=None)

    args = parser.parse_args(argv)
    if args.num_persistence_groups is not None and not 1 <= args.num_persistence_groups <= args.dict_size:
        parser.error(f"--num-persistence-groups must be between 1 and --dict-size ({args.dict_size}).")
    return args


def default_run_name(args: argparse.Namespace) -> str:
    model_tag = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    groups = f"_pg{args.num_persistence_groups}" if args.num_persistence_groups is not None else ""
    init = args.init_persistence if args.persistence_init == "constant" else args.persistence_init
    return f"{model_tag}_l{args.layer}_persistent_k{args.k}{groups}_ctx{args.ctx_len}_pinit{init}_s{args.max_steps}"


def build_save_steps(max_steps: int, start_step: int = 1000, every_steps: int | None = None) -> list[int]:
    steps = [x for x in (1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000) if start_step <= x <= max_steps]
    if every_steps is not None:
        steps.extend(range(every_steps, max_steps + 1, every_steps))
    return sorted({step for step in steps if start_step <= step < max_steps})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def capture_training_state(trainer: PersistentSAETrainer, next_step: int, buffer: SequenceActivationBuffer) -> dict:
    return {
        "format_version": TRAINING_STATE_FORMAT,
        "next_step": int(next_step),
        "ae": {k: v.cpu() for k, v in trainer.ae.state_dict().items()},
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "num_tokens_since_fired": trainer.num_tokens_since_fired.cpu(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "source_examples_consumed": int(buffer.data.examples_consumed),
        "trainer_config": trainer.config,
        "buffer_config": buffer.config,
    }


def restore_training_state(trainer: PersistentSAETrainer, state: dict) -> int:
    if int(state.get("format_version", -1)) != TRAINING_STATE_FORMAT:
        raise ValueError(f"Unsupported training-state format: {state.get('format_version')!r}")
    saved, current = state["trainer_config"], trainer.config
    mismatches = {
        key: (saved.get(key), current.get(key)) for key in RESUME_COMPATIBILITY_KEYS if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(f"Resume trainer configuration mismatch (saved, current): {mismatches}")
    trainer.ae.load_state_dict(state["ae"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.num_tokens_since_fired.copy_(state["num_tokens_since_fired"].to(trainer.num_tokens_since_fired.device))
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if state.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        # map_location moves the saved states onto the GPU; RNG setters need CPU ByteTensors.
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_state_all"]])
    return int(state["next_step"])


def save_checkpoint(trainer: PersistentSAETrainer, buffer, trainer_dir: Path, step: int, final: bool = False) -> None:
    weights = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
    training_state = capture_training_state(trainer, next_step=step, buffer=buffer)
    if final:
        torch.save(weights, trainer_dir / "ae.pt")
        atomic_torch_save(training_state, trainer_dir / "training_state.pt")
    else:
        checkpoints_dir = trainer_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        torch.save(weights, checkpoints_dir / f"ae_{step}.pt")
        atomic_torch_save(training_state, checkpoints_dir / f"training_state_{step}.pt")


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Persistent SAE training.")
    device = "cuda"

    resume_state = None
    skip_examples = 0
    if args.resume is not None:
        resume_state = torch.load(Path(args.resume).expanduser(), map_location=device, weights_only=False)
        skip_examples = int(resume_state.get("source_examples_consumed", 0))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    activation_dim = int(model.config.hidden_size)

    buffer = SequenceActivationBuffer(
        data=StreamingTextIterator(args.dataset, args.dataset_split, skip_examples=skip_examples),
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        d_submodule=activation_dim,
        device=device,
        n_ctxs=args.n_ctxs,
        ctx_len=args.ctx_len,
        out_batch_size=args.seq_batch_size,
        refresh_batch_size=args.refresh_batch_size,
    )
    trainer = PersistentSAETrainer(
        steps=args.max_steps,
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        layer=args.layer,
        lm_name=args.model,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        init_persistence=args.init_persistence,
        max_persistence=args.max_persistence,
        num_persistence_groups=args.num_persistence_groups,
        persistence_init=args.persistence_init,
        device=device,
    )
    start_step = 0
    if resume_state is not None:
        start_step = restore_training_state(trainer, resume_state)

    run_name = args.run_name or default_run_name(args)
    trainer_dir = Path(args.save_root) / run_name
    trainer_dir.mkdir(parents=True, exist_ok=True)
    with open(trainer_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"trainer": trainer.config, "buffer": buffer.config, "dataset": args.dataset}, f, indent=4)

    wandb_run = None
    if args.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={**trainer.config, **buffer.config, "dataset": args.dataset, "resumed_from": args.resume},
        )

    save_steps = set(build_save_steps(args.max_steps, args.checkpoint_start_step, args.checkpoint_every_steps))
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    for step in tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps, desc="Training"):
        x, token_mask = next(buffer)
        if step in save_steps:
            save_checkpoint(trainer, buffer, trainer_dir, step)
        log_now = args.log_steps > 0 and step % args.log_steps == 0
        with autocast:
            stats = trainer.update(step, x.to(torch.bfloat16), token_mask, return_stats=log_now)
        if stats is not None and wandb_run is not None:
            wandb_run.log(stats, step=step)

    save_checkpoint(trainer, buffer, trainer_dir, args.max_steps, final=True)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved {trainer_dir / 'ae.pt'}")


if __name__ == "__main__":
    main()

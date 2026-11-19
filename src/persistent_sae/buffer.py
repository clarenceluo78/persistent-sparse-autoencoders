"""Streaming text and contiguous-sequence activation buffers for Persistent SAE training."""

from __future__ import annotations

import torch as t
from datasets import load_dataset


class StreamingTextIterator:
    """Iterate over the non-empty ``text`` fields of a streamed Hugging Face dataset."""

    def __init__(self, dataset_name: str, split: str, skip_examples: int = 0):
        if skip_examples < 0:
            raise ValueError("skip_examples must be non-negative.")
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        if skip_examples:
            dataset = dataset.skip(int(skip_examples))
        self._iterator = iter(dataset)
        # Counted so that resumed runs can skip the documents already seen.
        self.examples_consumed = int(skip_examples)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        while True:
            example = next(self._iterator)
            self.examples_consumed += 1
            text = example.get("text", None)
            if text:
                return text


def get_decoder_layer(model, layer: int):
    """Return the transformer block whose output is the residual stream after ``layer``."""
    for path in ("gpt_neox.layers", "model.layers", "transformer.h"):
        module = model
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            continue
        return module[layer]
    raise ValueError(f"Could not locate decoder layers for {type(model).__name__}.")


class _StopForward(Exception):
    pass


@t.no_grad()
def capture_layer_output(model, layer_module, input_ids: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
    """Residual-stream output of ``layer_module``; the forward pass stops after that layer."""
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, (tuple, list)) else output
        raise _StopForward

    handle = layer_module.register_forward_hook(hook)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    except _StopForward:
        pass
    finally:
        handle.remove()
    return captured["hidden"]


class SequenceActivationBuffer:
    """Yield padded (batch, seq, d) blocks of activations from whole documents.

    Unlike token-shuffling SAE buffers, each row is a contiguous document prefix of
    up to ``ctx_len`` tokens so the persistent state can be carried across tokens.
    """

    def __init__(
        self,
        data,
        model,
        tokenizer,
        layer: int,
        d_submodule: int,
        device: str,
        n_ctxs: int,
        ctx_len: int,
        out_batch_size: int,
        refresh_batch_size: int,
        remove_first_token: bool = True,
    ):
        """
        Args:
            data: iterator over raw text documents.
            n_ctxs: number of sequences held in the buffer.
            ctx_len: maximum tokens per sequence, including the first token.
            out_batch_size: sequences per training batch.
            refresh_batch_size: documents per language-model forward pass.
            remove_first_token: drop the first position of each sequence, whose
                activation is an attention-sink outlier (and the BOS token for models
                that prepend one).
        """
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.layer_module = get_decoder_layer(model, layer)
        self.d_submodule = int(d_submodule)
        self.device = device
        self.n_ctxs = int(n_ctxs)
        self.ctx_len = int(ctx_len)
        self.out_batch_size = int(out_batch_size)
        self.refresh_batch_size = int(refresh_batch_size)
        self.remove_first_token = bool(remove_first_token)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.activations: list[t.Tensor] = []
        self.read = t.zeros(0, dtype=t.bool, device=self.device)

    @property
    def config(self) -> dict:
        return {
            "buffer_class": "SequenceActivationBuffer",
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "max_ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
            "remove_first_token": self.remove_first_token,
        }

    def __iter__(self):
        return self

    def __next__(self) -> tuple[t.Tensor, t.Tensor]:
        unread = (~self.read).nonzero(as_tuple=False).flatten()
        if unread.numel() < self.out_batch_size:
            self.refresh()
            unread = (~self.read).nonzero(as_tuple=False).flatten()
            if unread.numel() == 0:
                raise StopIteration

        if unread.numel() <= self.out_batch_size:
            idx = unread
        else:
            perm = t.randperm(unread.numel(), device=unread.device)[: self.out_batch_size]
            idx = unread[perm]

        self.read[idx] = True
        batch = [self.activations[int(i)] for i in idx.tolist()]
        max_len = max(seq.size(0) for seq in batch)
        padded = t.zeros((len(batch), max_len, self.d_submodule), dtype=batch[0].dtype, device=self.device)
        mask = t.zeros((len(batch), max_len), dtype=t.bool, device=self.device)
        for row, seq in enumerate(batch):
            padded[row, : seq.size(0)] = seq
            mask[row, : seq.size(0)] = True
        return padded, mask

    @t.no_grad()
    def refresh(self):
        seq_acts = [seq for seq, was_read in zip(self.activations, self.read.tolist()) if not was_read]

        while len(seq_acts) < self.n_ctxs:
            texts = [next(self.data) for _ in range(self.refresh_batch_size)]
            tokens = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=self.ctx_len
            ).to(self.model.device)
            hidden = capture_layer_output(self.model, self.layer_module, tokens["input_ids"], tokens["attention_mask"])

            # Select real tokens through the attention mask so either padding side works.
            valid = tokens["attention_mask"].bool()
            for row in range(hidden.size(0)):
                seq = hidden[row][valid[row]]
                if self.remove_first_token:
                    seq = seq[1:]
                if seq.size(0) == 0:
                    continue
                seq_acts.append(seq.to(self.device, non_blocking=True))
                if len(seq_acts) >= self.n_ctxs:
                    break

        self.activations = seq_acts
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

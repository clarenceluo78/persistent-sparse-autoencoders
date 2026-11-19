# Persistent Sparse Autoencoders

[![arXiv](https://img.shields.io/badge/arXiv-2607.17117-b31b1b.svg)](https://arxiv.org/abs/2607.17117) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-yellow)](https://huggingface.co/HF_REPO_PLACEHOLDER)

This repository has the code and trained checkpoints for the paper **[Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations](https://arxiv.org/abs/2607.17117)**.

## Installation

We recommend Python 3.10 or later and PyTorch 2.1 or later.

```bash
cd persistent-sparse-autoencoders
pip install -e .            # or ".[wandb]" for Weights & Biases logging
```

## Using Persistent SAE

Trained Persistent SAEs from our paper can be found on **[Hugging Face](https://huggingface.co/haoyanl78/persistent-sparse-autoencoders)**.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from persistent_sae import PersistentSAE

lm_name, layer = "EleutherAI/pythia-160m-deduped", 8
tokenizer = AutoTokenizer.from_pretrained(lm_name)
lm = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=torch.bfloat16).cuda().eval()
sae = PersistentSAE.from_pretrained("path/to/ae.pt", device="cuda")

inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to("cuda")
with torch.no_grad():
    hidden_states = lm(**inputs, output_hidden_states=True).hidden_states
    # drop the first token, as in training
    x = hidden_states[layer + 1][:, 1:].float()   # (batch, seq, d_model)

    x_hat, h = sae(x, output_features=True)       # reconstruction and persistent state h_t
    a = sae.encode_evidence(x)                    # per-token activation
    lam = sae.persistence()                       # learned λ for each feature, shape (dict_size,)
```

When a batch contains padding, pass `token_mask` (for example `inputs.attention_mask[:, 1:]`) to `sae(...)`, `encode_evidence` and `encode_state`.

## Training

To reproduce SAEs from the paper:

```bash
bash scripts/train_pythia160m_l8.sh    # Pythia-160M-deduped, layer 8
bash scripts/train_gemma2b_l12.sh      # Gemma-2-2B, layer 12
```

Both train 16,384 features with BatchTopK (k = 20) for 244,140 steps (about 500M tokens of The Pile). The final weights are saved to `checkpoints/<run-name>/ae.pt`.

### Configuration

The main options are below. Run `python -m persistent_sae.train --help` for details.

| Option | Default | Description |
| --- | --- | --- |
| `--model`, `--layer` | see scripts | Language model, and the decoder block whose output the SAE reconstructs. |
| `--dict-size` | 16384 | Number of features. |
| `--k` | 20 | Average number of active features per token (BatchTopK). |
| `--init-persistence` | 0.5 | Starting persistence λ for every feature. |
| `--num-persistence-groups` | unset | Share one λ within each of N feature groups. Unset means one λ per feature. |
| `--max-steps` | 244140 | Training steps. |
| `--lr` | 3e-4 | Learning rate. |
| `--refresh-batch-size` | 1024 / 512 | Documents per LM forward pass. Lower it if you run out of GPU memory. |
| `--wandb-project` | unset | Log to Weights & Biases. |
| `--resume` | unset | Resume from a `training_state*.pt` file. |

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this code or the checkpoints, please cite:

```bibtex
@misc{luo2026persistentsparseautoencoderslearning,
      title={Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations},
      author={Haoyan Luo and Mateo Espinosa Zarlenga and Mateja Jamnik},
      year={2026},
      eprint={2607.17117},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.17117},
}
```

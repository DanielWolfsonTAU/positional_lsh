# Positional LSH Attention

## Code Structure

This project consists of three GitHub repositories:

- **[`positional_lsh`](https://github.com/DanielWolfsonTAU/positional_lsh)** — This repository. Contains the core code for training and validating the theoretical results of the paper.

- **[`transformers`](https://github.com/DanielWolfsonTAU/transformers/tree/positional_lsh)** — Fork of the [HuggingFace Transformers](https://github.com/huggingface/transformers) library (based on `main` at **v4.56.1**). The modifications are on the **`positional_lsh`** branch.

- **[`flash-attention`](https://github.com/DanielWolfsonTAU/flash-attention/tree/positional_lsh)** — Fork of the [Dao-AILab Flash Attention](https://github.com/Dao-AILab/flash-attention) library (based on `main` at **v2.8.3**). The modifications are on the **`positional_lsh`** branch.

---

## Installation and Prerequisites

**Recommended Python version: 3.11**

Clone all three repositories:

```bash
git clone https://github.com/DanielWolfsonTAU/positional_lsh.git
git clone --branch positional_lsh https://github.com/DanielWolfsonTAU/transformers.git
git clone --branch positional_lsh https://github.com/DanielWolfsonTAU/flash-attention.git
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install the included modified `transformers` package (required for both sections):

```bash
cd transformers
pip install -e .
cd ..
```

For the runs of Section 1, the `flash-attention` sources must also be built and installed locally so that the Positional LSH kernel changes are applied. The build is controlled by the `FLASH_ATTENTION_LSH_MODE` environment variable:

```bash
cd flash-attention

# LSH mode (default) — disables ALiBi in the kernel; positional bias is handled in Python
FLASH_ATTENTION_LSH_MODE=TRUE python -m pip install -v --no-build-isolation -e .

# ALiBi mode — restores standard ALiBi computation inside the kernel
FLASH_ATTENTION_LSH_MODE=FALSE python -m pip install -v --no-build-isolation -e .

cd ..
```

### Environment Variables

The following environment variables must be set before running:

- `HF_TOKEN` — HuggingFace read token, required to download the models:
  - `Qwen/Qwen3-0.6B`
  - `mistralai/Mistral-7B-v0.3`
  - `meta-llama/Llama-4-Scout-17B-16E`

- `WANDB_API_KEY` — Weights & Biases API key, required for experiment logging during training (Section 1).

---

## Section 1 — Training

This section corresponds to the experiments in Section 5.1 of the paper.

The "main.py" script trains a causal language model with a choice of attention mechanism on WikiText-103.

The evaluation results are tracked and plotted in Weights & Biases.
### Usage

```bash
cd positional_lsh
python main.py [OPTIONS]
```

### Arguments

- `--model` *(str, default: `qwen3`)* — Model architecture to train. Choices: `qwen3`, `mistral`.
- `--attn_mode` *(str, default: `lsh`)* — Attention mode. Choices: `lsh`, `alibi`, `original`, `transfer` (see below).
- `--epochs` *(int, default: `5`)* — Number of training epochs.
- `--learning_rate` *(float, default: `5e-4` for qwen3, `1e-5` for mistral)* — Learning rate for Adam.
- `--weight_decay` *(float, default: `0.01`)* — Weight decay for Adam.
- `--seed` *(int, default: `42`)* — Random seed for dataset shuffling and training reproducibility.
- `--use_lora` *(flag, default: `False`)* — Use LoRA parameter-efficient fine-tuning.
- `--num_lsh_samples` *(int)* — Number of LSH hash draws (defaults to 100 for qwen3, 15 for mistral).
- `--fixed_block_size` *(flag, default: `False`)* — Fix block length and shift (b=2σ, c=0); by default blocks are drawn stochastically from Gamma(2, σ).
- `--sigma_constant` *(float, default: `256.0`)* — Scale parameter σ; expected block size = 2σ.

### Attention modes

- `lsh` — Positional LSH attention.
- `alibi` — ALiBi positional bias, no LSH.
- `original` — no ALiBi, no LSH; using the model's positional encoding method.
- `transfer` — LSH model initialised from pretrained model weights - used for LoRA.

---

## Section 2 — Theoretical Validation

This section corresponds to the experiments in Section 5.2 of the paper.

The "theory_validation.py" script validates the theoretical approximation quality of Positional LSH against ALiBi attention matrices. It runs forward passes on WikiText-103 inputs, extracts Q/K/V activations, and measures metrics across varying numbers of hash samples.
The script prints the results to stdout.

### Usage

```bash
cd positional_lsh
python theory_validation.py [OPTIONS]
```

### Arguments

- `--model` *(str, default: `llama4`)* — Model to extract activations from. Choices: `llama4`, `mistral`.
- `--debug` *(flag, default: `False`)* — Print detailed per-head matrix output.

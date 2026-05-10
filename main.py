import torch
import flash_attn_2_cuda
import flash_attn.flash_attn_interface
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "transformers" / "src"))

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLMEx
from transformers.models.mistral.modeling_mistral import MistralForCausalLMEx

from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen3ForCausalLM, Qwen3Config
from transformers import MistralConfig, MistralForCausalLM

from datasets import load_dataset
from torch.optim import AdamW
from transformers import TrainerCallback
import wandb
from torch.utils.data import DataLoader
import torch, random

import argparse
from datetime import datetime
import math
from functools import partial

from peft import LoraConfig, get_peft_model, TaskType

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # expose only one GPU to PyTorch
os.environ["NCCL_DEBUG"] = "WARN"             # optional, quieter logs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def compute_top_k_accuracy(logits, labels, k=1):
    """
    Counts correct top-k next-token predictions at valid (non-masked) positions.

    - logits: [batch, seq_len, vocab]
    - labels: [batch, seq_len], with -100 at positions to ignore
    - Returns the count of positions where the true token ranks within the top k.
    """
    valid_mask = labels.ne(-100)
    if valid_mask.sum().item() == 0:
        return 0

    topk = torch.topk(logits, k, dim=-1).indices
    correct = topk.eq(labels.unsqueeze(-1)).any(dim=-1)
    correct = correct & valid_mask
    return correct.sum().item()

def compute_mrr(logits, labels):
    """Sum of reciprocal ranks of the correct token at valid (non-masked) positions."""
    valid_mask = labels.ne(-100)
    if valid_mask.sum().item() == 0:
        return 0.0
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_labels = labels.reshape(-1)
    flat_mask = valid_mask.reshape(-1)
    valid_logits = flat_logits[flat_mask]
    valid_labels = flat_labels[flat_mask]
    correct_scores = valid_logits[torch.arange(len(valid_labels), device=valid_logits.device), valid_labels]
    ranks = (valid_logits > correct_scores.unsqueeze(-1)).sum(dim=-1) + 1
    return (1.0 / ranks.float()).sum().item()


def compute_mean_entropy(logits, labels):
    """Sum of per-token softmax entropy (nats) at valid (non-masked) positions."""
    valid_mask = labels.ne(-100)
    if valid_mask.sum().item() == 0:
        return 0.0
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_mask = valid_mask.reshape(-1)
    valid_logits = flat_logits[flat_mask]
    log_probs = torch.nn.functional.log_softmax(valid_logits, dim=-1)
    entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
    return entropy.sum().item()


def format_example_for_eval(example, tokenizer):
    """
    Converts one example into a text string for evaluation.

    Supported schemas:
    1) {"messages": [{"role": ..., "content": ...}, ...]}
    2) {"prompt": ..., "response": ...}
    3) {"instruction": ..., "output": ...}
    4) {"text": ...}

    For post-trained/chat models, chat-style formatting is preferred.
    """
    if "messages" in example and example["messages"] is not None:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if "prompt" in example and "response" in example:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    if "instruction" in example and "output" in example:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    if "text" in example and example["text"] is not None:
        return example["text"]

    raise ValueError(
        "Unsupported eval dataset schema. Expected one of: "
        "`messages`, `prompt`+`response`, `instruction`+`output`, or `text`."
    )


def concatenate_and_tokenize(examples, tokenizer, max_length):
    """
    Tokenizes a batch of examples into fixed-length non-overlapping chunks:
    - format each example into text
    - concatenate all texts
    - split into chunks of max_length tokens
    - pad the final chunk if needed
    """
    texts = []

    batch_size = len(next(iter(examples.values())))
    for i in range(batch_size):
        example_i = {k: examples[k][i] for k in examples.keys()}
        text = format_example_for_eval(example_i, tokenizer)
        if text is not None and len(text.strip()) > 0:
            texts.append(text)

    if len(texts) == 0:
        return {
            "input_ids": [],
            "attention_mask": [],
        }

    concatenated_text = "\n\n".join(texts)

    tokenized = tokenizer(
        concatenated_text,
        add_special_tokens=False,
        return_attention_mask=True,
        truncation=False,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    chunked_input_ids = []
    chunked_attention_mask = []

    for i in range(0, len(input_ids), max_length):
        chunk_ids = input_ids[i:i + max_length]
        chunk_mask = attention_mask[i:i + max_length]

        pad_len = max_length - len(chunk_ids)
        if pad_len > 0:
            chunk_ids = chunk_ids + [tokenizer.pad_token_id] * pad_len
            chunk_mask = chunk_mask + [0] * pad_len

        chunked_input_ids.append(chunk_ids)
        chunked_attention_mask.append(chunk_mask)

    return {
        "input_ids": chunked_input_ids,
        "attention_mask": chunked_attention_mask,
    }

def add_labels_and_mask(example):
    """
    Creates labels from input_ids and masks padding tokens with -100
    so they are ignored by the loss.
    """
    input_ids = example["input_ids"]
    attention_mask = example.get("attention_mask", [1] * len(input_ids))

    labels = [
        token_id if mask_value == 1 else -100
        for token_id, mask_value in zip(input_ids, attention_mask)
    ]

    return {"labels": labels}


class MetricsCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that computes and logs evaluation metrics to WandB.

    Metrics reported at each evaluation step:
    - cross-entropy loss and perplexity
    - bits per token
    - top-1 / top-10 / top-100 next-token accuracy
    - mean reciprocal rank (MRR)
    - mean predictive entropy
    """

    def __init__(self, trainer=None):
        """Initialize the callback; trainer is injected after Trainer construction."""
        super().__init__()
        self.trainer = trainer

    @staticmethod
    def run_eval_loop(model, dataloader, device, start_pos=0):
        """
        Iterates over dataloader and computes aggregated evaluation metrics.

        - When start_pos > 0, only tokens at positions >= start_pos are scored,
          supporting length-extrapolation evaluation.
        - Returns a dict with: eval_loss, perplexity, bits_per_token,
          top1/10/100_accuracy, mrr, mean_entropy.
        """
        model.eval()

        total_nll = 0.0
        total_valid_tokens = 0

        total_correct_top1 = 0
        total_correct_top10 = 0
        total_correct_top100 = 0
        total_mrr = 0.0
        total_entropy = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                logits = outputs.logits
                labels = batch["labels"]

                # Standard causal LM shifting
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # For length extrapolation: only evaluate tokens beyond start_pos
                if start_pos > 0 and shift_labels.shape[1] > start_pos:
                    shift_logits = shift_logits[:, start_pos:, :]
                    shift_labels = shift_labels[:, start_pos:]

                # Sum loss over valid tokens only
                loss_sum = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                valid_mask = shift_labels.ne(-100)
                num_valid_tokens = valid_mask.sum().item()

                if num_valid_tokens == 0:
                    continue

                total_nll += loss_sum.item()
                total_valid_tokens += num_valid_tokens

                total_correct_top1 += compute_top_k_accuracy(shift_logits, shift_labels, k=1)
                total_correct_top10 += compute_top_k_accuracy(shift_logits, shift_labels, k=10)
                total_correct_top100 += compute_top_k_accuracy(shift_logits, shift_labels, k=100)
                total_mrr += compute_mrr(shift_logits, shift_labels)
                total_entropy += compute_mean_entropy(shift_logits, shift_labels)

        if total_valid_tokens == 0:
            raise ValueError("No valid evaluation tokens found. Check label creation and masking.")

        avg_loss = total_nll / total_valid_tokens
        perplexity = math.exp(avg_loss)
        bits_per_token = avg_loss / math.log(2)

        top1_accuracy = 100.0 * total_correct_top1 / total_valid_tokens
        top10_accuracy = 100.0 * total_correct_top10 / total_valid_tokens
        top100_accuracy = 100.0 * total_correct_top100 / total_valid_tokens
        mrr = total_mrr / total_valid_tokens
        mean_entropy = total_entropy / total_valid_tokens

        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "bits_per_token": bits_per_token,
            "top1_accuracy": top1_accuracy,
            "top10_accuracy": top10_accuracy,
            "top100_accuracy": top100_accuracy,
            "mrr": mrr,
            "mean_entropy": mean_entropy,
        }

    def on_evaluate(self, args, state, control, **kwargs):
        """Called by the Trainer after each evaluation step; logs metrics to WandB."""
        trainer = self.trainer
        model = trainer.model
        dataloader = trainer.get_eval_dataloader()

        metrics = self.run_eval_loop(model, dataloader, trainer.args.device)

        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated(trainer.args.device) / (1024 ** 2)
            gpu_reserved = torch.cuda.memory_reserved(trainer.args.device) / (1024 ** 2)
        else:
            gpu_allocated = 0.0
            gpu_reserved = 0.0

        wandb.log({
            "eval_loss": metrics["eval_loss"],
            "perplexity": metrics["perplexity"],
            "bits_per_token": metrics["bits_per_token"],
            "top1_accuracy": metrics["top1_accuracy"],
            "top10_accuracy": metrics["top10_accuracy"],
            "top100_accuracy": metrics["top100_accuracy"],
            "mrr": metrics["mrr"],
            "mean_entropy": metrics["mean_entropy"],
            "gpu_allocated_MB": gpu_allocated,
            "gpu_reserved_MB": gpu_reserved,
        })

        print(
            f"[Eval] loss: {metrics['eval_loss']:.4f}  "
            f"ppl: {metrics['perplexity']:.2f}  "
            f"bpt: {metrics['bits_per_token']:.4f}  "
            f"top-1: {metrics['top1_accuracy']:.2f}%  "
            f"top-10: {metrics['top10_accuracy']:.2f}%  "
            f"mrr: {metrics['mrr']:.4f}  "
            f"entropy: {metrics['mean_entropy']:.4f}  "
            f"gpu_alloc: {gpu_allocated:.0f} MB"
        )


def evaluate(model, tokenizer, eval_dataset, max_length, trainer=None, tokenize_fn=None, start_pos=0, stride=None):
    """
    Evaluates the model on eval_dataset and returns aggregated metrics.

    - tokenize_fn: if provided, used to tokenize the dataset; otherwise
      concatenate_and_tokenize is used.
    - stride: when set, uses overlapping windows of width max_length, scoring
      only the tail tokens at positions [max_length - stride : max_length],
      following the sliding-window perplexity protocol.
    - Returns the same metric dict as MetricsCallback.run_eval_loop.
    """
    model.eval()

    if tokenize_fn is not None:
        if stride is not None:
            tokenize_with_max_length = partial(tokenize_fn, max_length=max_length, stride=stride)
        else:
            tokenize_with_max_length = partial(tokenize_fn, max_length=max_length)
    else:
        tokenize_with_max_length = partial(
            concatenate_and_tokenize,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    # Remove whatever columns exist in the source dataset, not just ["text"]
    remove_columns = eval_dataset.column_names

    eval_dataset = eval_dataset.map(
        tokenize_with_max_length,
        batched=True,
        remove_columns=remove_columns,
    )

    if "labels" not in eval_dataset.column_names:
        eval_dataset = eval_dataset.map(add_labels_and_mask)

    eval_dataset.set_format(type="torch")

    if trainer is not None:
        dataloader = DataLoader(
            eval_dataset,
            batch_size=trainer.args.per_device_eval_batch_size,
            collate_fn=trainer.data_collator,
        )
        device = trainer.args.device
    else:
        dataloader = DataLoader(eval_dataset, batch_size=1)
        device = next(model.parameters()).device

    # Standard sliding-window perplexity (matches ALiBi paper):
    # score the last `stride` tokens of each window — positions [max_length - stride : max_length].
    # This gives a constant number of scored tokens per window regardless of max_length,
    # so results are directly comparable across eval lengths.
    effective_start_pos = (max_length - stride) if stride is not None else start_pos
    metrics = MetricsCallback.run_eval_loop(model, dataloader, device, start_pos=effective_start_pos)

    return {
        "eval_loss": metrics["eval_loss"],
        "perplexity": metrics["perplexity"],
        "bits_per_token": metrics["bits_per_token"],
        "top1_accuracy": metrics["top1_accuracy"],
        "top10_accuracy": metrics["top10_accuracy"],
        "top100_accuracy": metrics["top100_accuracy"],
        "mrr": metrics["mrr"],
        "mean_entropy": metrics["mean_entropy"],
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help="Number of training epochs"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="The seed to use for training"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen3',
        choices=['qwen3', 'mistral'],
        help="The model to use"
    )
    
    temp_args, _ = parser.parse_known_args()

    # Set the default learning rate based on the model
    if temp_args.model == 'qwen3':
        default_lr = 5e-4
    else:
        default_lr = 1e-5
        
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=default_lr,
        help="Learning rate for the AdamW optimizer"
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help="Weight decay for the AdamW optimizer"
    )
    
    parser.add_argument(
        '--attn_mode',
        type=str,
        default='lsh',
        choices=['lsh', 'alibi', 'transfer', 'original'],
        help="lsh: LSH attention; alibi: ALiBi attention (no LSH); original: no positional bias (no ALiBi, no LSH);"
             " transfer: LSH model init from pretrained alibi model weights"
    )
    
    parser.add_argument(
        '--use_lora',
        action='store_true',
        default=False,
        help="use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        '--num_lsh_samples',
        type=int,
        default=None,
        help="number of independent LSH hash draws; output is their mean. Default: 100 for qwen3, 15 for mistral"
    )
    parser.add_argument(
        '--fixed_block_size',
        action='store_true',
        default=False,
        help="if set, fix block length and shift (b=2*sigma, c=0); "
             "default is stochastic: b ~ Gamma(2, sigma), c ~ Uniform(0, b)"
    )
    parser.add_argument(
        '--sigma_constant',
        type=float,
        default=256.0,
        help="scale parameter sigma; expected block size = 2 * sigma"
    )

    # Parse the arguments
    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    seed = args.seed
    model_name = args.model
    weight_decay = args.weight_decay
    attn_mode = args.attn_mode
    use_lora = args.use_lora
    num_lsh_samples_qwen    = args.num_lsh_samples if args.num_lsh_samples is not None else 100
    num_lsh_samples_mistral = args.num_lsh_samples if args.num_lsh_samples is not None else 15
    fixed_block_size = args.fixed_block_size
    sigma_constant = args.sigma_constant
    
    
    # Initialize Weights and Biases (WandB)
    wandb.init(project="positional-lsh")
    if model_name == 'mistral':

        # Patch transformers parallel style registry to include all required styles.
        try:
            from torch.distributed.tensor import DTensor  # noqa: F401
        except ImportError:
            pass
        from transformers import modeling_utils

        SUPPORTED = (
            "none", "tp", "dp", "fsdp", "deepspeed", "pipeline",
            "colwise", "rowwise", "colwise_rep", "rowwise_rep",
            "local_colwise", "local_rowwise", "local",
            "gather", "local_packed_rowwise",
            "sequence_parallel", "replicate",
        )
        if not getattr(modeling_utils, "ALL_PARALLEL_STYLES", None):
            modeling_utils.ALL_PARALLEL_STYLES = SUPPORTED
        else:
            modeling_utils.ALL_PARALLEL_STYLES = tuple(
                sorted(set(modeling_utils.ALL_PARALLEL_STYLES) | set(SUPPORTED))
            )


        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", 0)))
        global_rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", 0)))
        torch.cuda.set_device(local_rank)
        main_rank = global_rank == 0
        if not main_rank:
            os.environ["WANDB_MODE"] = "disabled"  # must be set before wandb is imported

        import wandb

        if main_rank:  # only rank-0 creates a run
            wandb.init(project="positional-lsh")


        # Load and patch tokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        tokenizer.model_max_length = int(1e12)  # suppress "longer than max_length" warnings

        _mistral_ckpt = "mistralai/Mistral-7B-v0.3"
        _mistral_config = MistralConfig.from_pretrained(_mistral_ckpt)
        _mistral_config.use_cache = False
        _mistral_config._attn_implementation = "flash_attention_2"
        _mistral_config.use_lsh = (attn_mode in ("lsh", "transfer"))
        _mistral_config.num_lsh_samples = num_lsh_samples_mistral
        _mistral_config.fixed_block_size = fixed_block_size
        _mistral_config.sigma_constant = sigma_constant

        if use_lora:
            print("Using LoRA for fine-tuning.")

            base_model = MistralForCausalLM.from_pretrained(
                _mistral_ckpt,
                config=_mistral_config,
                torch_dtype=torch.bfloat16,
            )

            if attn_mode in ("lsh", "alibi", "transfer"):
                print(f"Attention mode: {attn_mode}.")
                model = MistralForCausalLMEx(_mistral_config)
            else:
                print("Attention mode: original (no positional bias).")
                model = MistralForCausalLM(_mistral_config)

            missing_keys, unexpected_keys = model.load_state_dict(base_model.state_dict(), strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            del base_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            model.to("cuda", dtype=torch.bfloat16)

        else:
            # No LoRA — load the appropriate architecture directly.
            if attn_mode in ("lsh", "alibi", "transfer"):
                model = MistralForCausalLMEx.from_pretrained(
                    _mistral_ckpt,
                    config=_mistral_config,
                    torch_dtype=torch.bfloat16,
                )

            else:
                model = MistralForCausalLM.from_pretrained(
                    _mistral_ckpt,
                    config=_mistral_config,
                    torch_dtype=torch.bfloat16,
                )

        # define a real pad token (only once per tokenizer dir)
        if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))  # important after adding tokens

        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")
        assert tokenizer.pad_token_id != tokenizer.eos_token_id

        pad_token_id = tokenizer.pad_token_id
        max_length = 4096

        _DATASET = "wikitext"
        _DATASET_CONFIG = "wikitext-103-raw-v1"
        _SEED = 42

        print(f"[DATA] Loading {_DATASET}/{_DATASET_CONFIG} ...")
        _ds = load_dataset(_DATASET, _DATASET_CONFIG)
        train_raw = _ds["train"].shuffle(seed=_SEED).filter(lambda e: bool(e["text"] and e["text"].strip()))
        eval_raw = _ds["validation"].shuffle(seed=_SEED).filter(lambda e: bool(e["text"] and e["text"].strip()))


        def safe_concatenate_and_tokenize(examples, max_length, stride=None):
            if stride is None:
                stride = max_length
            concatenated_text = " ".join(examples["text"])
            tokenized_output = tokenizer(concatenated_text, truncation=False, padding=False)
            input_ids = tokenized_output["input_ids"]
            attention_mask = tokenized_output["attention_mask"]
            tokenized_batches = {"input_ids": [], "attention_mask": [], "labels": []}
            for i in range(0, len(input_ids) - max_length + 1, stride):
                chunk_input_ids = input_ids[i:i + max_length]
                chunk_attention_mask = attention_mask[i:i + max_length]
                chunk_labels = [tok if mask == 1 else -100 for tok, mask in zip(chunk_input_ids, chunk_attention_mask)]
                tokenized_batches["input_ids"].append(chunk_input_ids)
                tokenized_batches["attention_mask"].append(chunk_attention_mask)
                tokenized_batches["labels"].append(chunk_labels)
            if stride == max_length:
                last_start = (len(input_ids) // max_length) * max_length
                if last_start < len(input_ids):
                    chunk_input_ids = input_ids[last_start:]
                    chunk_attention_mask = attention_mask[last_start:]
                    _pad_len = max_length - len(chunk_input_ids)
                    chunk_input_ids += [pad_token_id] * _pad_len
                    chunk_attention_mask += [0] * _pad_len
                    chunk_labels = [tok if mask == 1 else -100 for tok, mask in zip(chunk_input_ids, chunk_attention_mask)]
                    tokenized_batches["input_ids"].append(chunk_input_ids)
                    tokenized_batches["attention_mask"].append(chunk_attention_mask)
                    tokenized_batches["labels"].append(chunk_labels)
            return tokenized_batches

        _tokenize_fn = partial(safe_concatenate_and_tokenize, max_length=max_length)

        print("[DATA] Tokenizing train ...")
        train_dataset = train_raw.map(_tokenize_fn, batched=True, remove_columns=["text"])
        print("[DATA] Tokenizing eval ...")
        eval_dataset = eval_raw.map(_tokenize_fn, batched=True, remove_columns=["text"])

        data_collator = default_data_collator


    if model_name == 'qwen3':

        base_ckpt = "Qwen/Qwen3-0.6B"

        max_length = 8192

        print(f"Attention mode: {attn_mode}")
        print(f"Qwen checkpoint: {base_ckpt}")
        print(f"Qwen max length: {max_length}\n")

        # -----------------------------
        # Tokenizer
        # -----------------------------
        tokenizer = AutoTokenizer.from_pretrained(base_ckpt)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        pad_token_id = tokenizer.pad_token_id
        actual_max_id = max(tokenizer.get_vocab().values())

        print(f"tokenizer.pad_token: {tokenizer.pad_token}")
        print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
        print(f"tokenizer.eos_token: {tokenizer.eos_token}")
        print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
        print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
        print(f"len(tokenizer): {len(tokenizer)}")
        print(f"actual_max_id: {actual_max_id}")

        assert pad_token_id is not None, "pad_token_id is None"
        assert pad_token_id <= actual_max_id, \
            f"pad_token_id {pad_token_id} is out of bounds! actual_max_id={actual_max_id}"

        # -----------------------------
        # Config
        # -----------------------------
        config = Qwen3Config.from_pretrained(base_ckpt)
        config.pad_token_id = pad_token_id
        config._attn_implementation = "flash_attention_2"
        config.attn_implementation = "flash_attention_2"
        config.use_lsh = (attn_mode in ("lsh", "transfer"))
        config.num_lsh_samples = num_lsh_samples_qwen
        config.fixed_block_size = fixed_block_size
        config.sigma_constant = sigma_constant

        # -----------------------------
        # Model
        # -----------------------------
        if attn_mode in ("lsh", "alibi", "transfer"):
            print(f"Initializing {attn_mode} attention model.")
            model = Qwen3ForCausalLMEx(config)
        else:
            print("Initializing model with no positional bias.")
            model = Qwen3ForCausalLM(config)

        model.to(device="cuda", dtype=torch.bfloat16)
        model.config.pad_token_id = pad_token_id
        model.config.use_cache = False

        # -----------------------------
        # LoRA
        # -----------------------------
        if use_lora:
            print("Using LoRA for fine-tuning.")

            base_model = Qwen3ForCausalLM.from_pretrained(
                base_ckpt,
                config=config,
                torch_dtype=torch.bfloat16,
            )

            if attn_mode in ("lsh", "alibi", "transfer"):
                model = Qwen3ForCausalLMEx(config)
            else:
                model = Qwen3ForCausalLM(config)

            missing_keys, unexpected_keys = model.load_state_dict(base_model.state_dict(), strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            del base_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            model.to("cuda", dtype=torch.bfloat16)


        # -----------------------------
        # Dataset
        # -----------------------------
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        orig_train_dataset = dataset["train"].shuffle(seed=42)
        orig_eval_dataset = dataset["validation"].shuffle(seed=42)

        print(f"Learning rate: {learning_rate}, seed: {seed}, model: {model_name}")
        print(f"Train set size: {len(orig_train_dataset)}, validation set size: {len(orig_eval_dataset)}")


        # -----------------------------
        # Tokenization with masked labels
        # -----------------------------
        def safe_concatenate_and_tokenize(examples, max_length, stride=None):
            # stride < max_length → overlapping sliding windows (for extrapolation eval).
            # stride == max_length (default) → original non-overlapping chunks.
            if stride is None:
                stride = max_length

            concatenated_text = " ".join(examples["text"])
            tokenized_output = tokenizer(
                concatenated_text,
                truncation=False,
                padding=False,
            )

            input_ids = tokenized_output["input_ids"]
            attention_mask = tokenized_output["attention_mask"]

            tokenized_batches = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

            # Full windows only — avoids partial chunks at boundaries which would
            # need padding and would corrupt start_pos-based tail scoring.
            for i in range(0, len(input_ids) - max_length + 1, stride):
                chunk_input_ids = input_ids[i:i + max_length]
                chunk_attention_mask = attention_mask[i:i + max_length]

                # Important: mask padded tokens for causal LM loss
                chunk_labels = [
                    tok if mask == 1 else -100
                    for tok, mask in zip(chunk_input_ids, chunk_attention_mask)
                ]

                tokenized_batches["input_ids"].append(chunk_input_ids)
                tokenized_batches["attention_mask"].append(chunk_attention_mask)
                tokenized_batches["labels"].append(chunk_labels)

            # In non-overlapping mode, pad and include the last partial chunk
            # (matches original behaviour).
            if stride == max_length:
                last_start = (len(input_ids) // max_length) * max_length
                if last_start < len(input_ids):
                    chunk_input_ids = input_ids[last_start:]
                    chunk_attention_mask = attention_mask[last_start:]
                    pad_len = max_length - len(chunk_input_ids)
                    chunk_input_ids += [pad_token_id] * pad_len
                    chunk_attention_mask += [0] * pad_len
                    chunk_labels = [
                        tok if mask == 1 else -100
                        for tok, mask in zip(chunk_input_ids, chunk_attention_mask)
                    ]
                    tokenized_batches["input_ids"].append(chunk_input_ids)
                    tokenized_batches["attention_mask"].append(chunk_attention_mask)
                    tokenized_batches["labels"].append(chunk_labels)

            return tokenized_batches


        tokenize_with_max_length = partial(safe_concatenate_and_tokenize, max_length=max_length)
        print("[DATA] Tokenizing train ...")
        train_dataset = orig_train_dataset.map(tokenize_with_max_length, batched=True, remove_columns=["text"])
        print("[DATA] Tokenizing eval ...")
        eval_dataset = orig_eval_dataset.map(tokenize_with_max_length, batched=True, remove_columns=["text"])
        
        # -----------------------------
        # Collator
        # -----------------------------
        # Since labels are already prepared, default_data_collator is safer here.
        data_collator = default_data_collator


    _script_dir = Path(__file__).resolve().parent
    _run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(_script_dir / f"positional_lsh_{model_name}_{attn_mode}_lr{learning_rate}_{_run_id}")
    os.environ.setdefault("WANDB_DIR", str(_script_dir))
    print(f"Output directory: {output_dir}")
    training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=learning_rate,
                seed=seed,
                weight_decay=weight_decay,
                warmup_ratio=0.1,
                bf16=True,
                fp16=False,
                load_best_model_at_end=True,
                remove_unused_columns=False,
                report_to="wandb",
                save_total_limit=1,
                save_only_model=True,
            )

    # Define the Adam optimizer with specific betas and epsilon
    if use_lora:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08
        )
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    
    if model_name == 'qwen3' or model_name == 'mistral':
        print("Starting training.")
        metrics_callback = MetricsCallback()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[metrics_callback],
            optimizers=(optimizer, None),
        )
        metrics_callback.trainer = trainer


    trainer.train()

    if model_name == 'qwen3':
        eval_lengths = [8192, 10000, 16384]
        training_length = max_length

        for length in eval_lengths:
            if length == training_length:
                # Baseline: regular non-overlapping eval, score every token.
                result = evaluate(model, tokenizer, orig_eval_dataset, length,
                                  tokenize_fn=safe_concatenate_and_tokenize, start_pos=0)
            else:
                # Extrapolation: overlapping windows stepped by training_length,
                # score only the tail tokens each window adds beyond training context.
                result = evaluate(model, tokenizer, orig_eval_dataset, length,
                                  tokenize_fn=safe_concatenate_and_tokenize,
                                  stride=training_length)

            val_loss = result["eval_loss"]
            val_perplexity = result["perplexity"]
            bits_per_token = result["bits_per_token"]
            top1_accuracy = result["top1_accuracy"]
            top10_accuracy = result["top10_accuracy"]
            top100_accuracy = result["top100_accuracy"]
            mrr = result["mrr"]
            mean_entropy = result["mean_entropy"]
            scored_range = f"0:{length}" if length == training_length else f"{length - training_length}:{length}"

            print(
                f"Max Length: {length}, Scored tokens: [{scored_range}], "
                f"Validation Loss: {val_loss}, Perplexity: {val_perplexity}, "
                f"BPT: {bits_per_token}, "
                f"top-1 accuracy:{top1_accuracy}, top-10 accuracy:{top10_accuracy}, "
                f"top-100 accuracy:{top100_accuracy}, MRR: {mrr}, Entropy: {mean_entropy}"
            )

            wandb.log({
                "Max Length": length,
                "Validation Loss": val_loss,
                "Perplexity": val_perplexity,
                "Bits Per Token": bits_per_token,
                "Top-1 Accuracy": top1_accuracy,
                "Top-10 Accuracy": top10_accuracy,
                "Top-100 Accuracy": top100_accuracy,
                "MRR": mrr,
                "Mean Entropy": mean_entropy,
            })


    elif model_name == 'mistral':
        training_length = max_length
        mistral_eval_lengths = [training_length, 8192, 10000, 16384]
        for length in mistral_eval_lengths:
            if length == training_length:
                result = evaluate(model, tokenizer, eval_raw, length,
                                  tokenize_fn=safe_concatenate_and_tokenize,
                                  start_pos=0)
            else:
                result = evaluate(model, tokenizer, eval_raw, length,
                                  tokenize_fn=safe_concatenate_and_tokenize,
                                  stride=training_length)
            val_loss = result["eval_loss"]
            val_perplexity = result["perplexity"]
            bits_per_token = result["bits_per_token"]
            top1_accuracy = result["top1_accuracy"]
            top10_accuracy = result["top10_accuracy"]
            top100_accuracy = result["top100_accuracy"]
            mrr = result["mrr"]
            mean_entropy = result["mean_entropy"]
            scored_range = f"0:{length}" if length == training_length else f"{length - training_length}:{length}"
            print(
                f"Max Length: {length}, Scored tokens: [{scored_range}], "
                f"Validation Loss: {val_loss}, Perplexity: {val_perplexity}, "
                f"BPT: {bits_per_token}, "
                f"top-1 accuracy:{top1_accuracy}, top-10 accuracy:{top10_accuracy}, "
                f"top-100 accuracy:{top100_accuracy}, MRR: {mrr}, Entropy: {mean_entropy}"
            )
            wandb.log({
                "Max Length": length,
                "Validation Loss": val_loss,
                "Perplexity": val_perplexity,
                "Bits Per Token": bits_per_token,
                "Top-1 Accuracy": top1_accuracy,
                "Top-10 Accuracy": top10_accuracy,
                "Top-100 Accuracy": top100_accuracy,
                "MRR": mrr,
                "Mean Entropy": mean_entropy,
            })

    print(f"Run complete. Output directory: {output_dir}")


    # Finish the WandB run
    wandb.finish()
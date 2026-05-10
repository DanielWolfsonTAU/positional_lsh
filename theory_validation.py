import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--model", type=str, default="llama4", choices=["llama4", "mistral"])
args = parser.parse_args()

DEBUG_TEST = args.debug
model_name = args.model

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

import math
from positional_lsh import  sample_hashes, generate_block_diagonal_masks_from_hashes
# -----------------------------
# ALIBI
# -----------------------------
def _get_alibi_slopes(n_heads):
    """
    Computes the slopes for ALiBi attention, designed for models with power-of-2 or arbitrary attention heads.

    Args:
    - n_heads: The number of attention heads for which slopes are computed.

    Returns:
    - slopes: A tensor of computed slopes for ALiBi attention, with device compatibility.
    """

    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
        slopes.extend(extra_slopes[: n_heads - closest_power_of_2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(slopes, device=device)


def _get_alibi_matrix(n: int, sigma: float, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    L[i, j] = exp(-|i - j| / sigma)
    """
    idx = torch.arange(n, device=device, dtype=dtype)
    dist = (idx[:, None] - idx[None, :]).abs()  # [n, n]
    return torch.exp(-dist / sigma)

def compute_Psi_sigma(sigma: float) -> float:
    """
    Ψσ = 1+ σ + (e^(1/σ) + 1) / (e^(1/σ) - 1)
    """
    return 1+ sigma + (math.exp(1/sigma) + 1) / (math.exp(1/sigma) - 1)

def bound_expectation(sigma: float, n: int, s: int) -> float:
    """New formula: Ψσ * (sqrt(log(n)/s) + log(n)/s)"""
    psi = compute_Psi_sigma(sigma) 
    logn_over_s = math.log(n) / s
    return psi * (math.sqrt(logn_over_s) + logn_over_s)

def _compute_Psi_sigma_explicit(sigma: float) -> float:
    e = math.exp(1 / sigma)
    return sigma + (e + 1) / (e - 1)


def bound_tail(sigma: float, n: int, s: int, epsilon: float) -> float:
    """
    Compute n * exp(- s * ε^2 / (4 Ψσ (4Ψσ + ε)))
    """
    psi = compute_Psi_sigma(sigma)
    exponent = - (s * (epsilon ** 2)) / (4 * psi * (4 * psi + epsilon))
    return n * math.exp(exponent)


# -----------------------------
# Models
# -----------------------------
if model_name == "llama4":
    model_id = "meta-llama/Llama-4-Scout-17B-16E"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Llama4ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
else:  # mistral
    model_id = "mistralai/Mistral-7B-v0.3"
    processor = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
model.eval()

# -----------------------------
# Load WikiText-103 and build 30 inputs of 4-5k tokens
# -----------------------------
from datasets import load_dataset

NUM_INPUTS = 30
TARGET_TOKENS = 4500  # target context length per input

print("Loading WikiText-103...")
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Concatenate all non-empty, non-header lines into one large text corpus
raw_text = " ".join(
    line.strip()
    for line in wiki["text"]
    if line.strip() and not line.strip().startswith("=")
)

def build_inputs_list(processor, raw_text, num_inputs, target_tokens, model_name):
    """Split corpus by words, tokenize each chunk individually to ~target_tokens."""
    words = raw_text.split()
    # estimate words per chunk: ~0.75 words per token
    words_per_chunk = int(target_tokens * 0.75)
    stride = max(words_per_chunk, len(words) // (num_inputs + 1))

    inputs_list = []
    for i in range(num_inputs):
        start = i * stride
        chunk_text = " ".join(words[start: start + words_per_chunk])

        if model_name == "llama4":
            inp = processor.tokenizer(chunk_text, return_tensors="pt")
        else:
            inp = processor(chunk_text, return_tensors="pt")

        actual_len = inp["input_ids"].shape[1]
        print(f"  Input {i+1}/{num_inputs}: {actual_len} tokens")
        inputs_list.append(inp)
    return inputs_list

print(f"Building {NUM_INPUTS} inputs of ~{TARGET_TOKENS} tokens each...")
inputs_list = build_inputs_list(processor, raw_text, NUM_INPUTS, TARGET_TOKENS, model_name)

# -----------------------------
# Extract Q/K/V via full model forward pass with hooks (all layers)
# -----------------------------
def make_proj_hook(kind, attn_module, layer_idx, storage):
    def hook(mod, inp, out):
        bsz, seqlen, _ = out.shape
        head_dim = attn_module.head_dim
        heads = out.shape[-1] // head_dim
        x = out.view(bsz, seqlen, heads, head_dim).permute(0, 2, 1, 3).contiguous()
        while len(storage) <= layer_idx:
            storage.append({"q": None, "k": None, "v": None})
        storage[layer_idx][kind] = x.detach().cpu()
    return hook

layers = model.language_model.model.layers if model_name == "llama4" else model.model.layers
_attn0 = layers[0].self_attn
num_heads = getattr(_attn0, "num_attention_heads", None) or getattr(_attn0, "config", None) and _attn0.config.num_attention_heads or _attn0.q_proj.weight.shape[0] // _attn0.head_dim
model_device = next(model.parameters()).device
all_captures = []
for inp_idx, inputs in enumerate(inputs_list):
    print(f"Forward pass {inp_idx+1}/{NUM_INPUTS}...")
    captured_qkv = []
    hooks = []
    for i, layer in enumerate(layers):  # all layers
        attn = layer.self_attn
        hooks.append(attn.q_proj.register_forward_hook(make_proj_hook("q", attn, i, captured_qkv)))
        hooks.append(attn.k_proj.register_forward_hook(make_proj_hook("k", attn, i, captured_qkv)))
        hooks.append(attn.v_proj.register_forward_hook(make_proj_hook("v", attn, i, captured_qkv)))
    inputs_on_device = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()}
    seq_len = inputs_on_device["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len} tokens  |  GPU memory before: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    with torch.no_grad():
        _ = model(**inputs_on_device, use_cache=False)
    print(f"  GPU memory after: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()
    all_captures.append(captured_qkv)

# Free the model from GPU to reclaim memory for matrix computations
del model
torch.cuda.empty_cache()
print("Model freed from GPU. Running metric computations on GPU.")

# -----------------------------
# Positional LSH - Q,K,V (aggregated over all inputs and heads)
# -----------------------------
s_values = [1, 10, 100, 1000, 10000, 100000]

table_metrics = {hq: {s: {"diff_norm": [], "diff_max_norm": [], "epsilon_emp": [], "diff_norm_exp": [],
                           "diff_max_exp": [], "diff_norm_sm": [], "rel_norm": [], "row_med": [],
                           "cos": [], "s_good": [], "exp_bound": [], "tail_bound": []}
                       for s in s_values}
                 for hq in range(num_heads)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alibi_slopes = _get_alibi_slopes(num_heads).to(device)
print(f"num_heads={num_heads}")
print("alibi_slopes (shape={}):\n{}".format(tuple(alibi_slopes.shape), alibi_slopes))

b = 0 
for inp_idx, captured_qkv in enumerate(all_captures):
    print(f"\n=== Input {inp_idx+1}/{NUM_INPUTS} ===")
    for layer_idx, qkv in enumerate(captured_qkv):
        q, k, v = qkv["q"], qkv["k"], qkv["v"]
        if q is None or k is None or v is None:
            continue

        n = q.shape[2]
        logn = math.log(n)
        print(f"\n  Layer {layer_idx}  n={n}")

        for hq, m in zip(range(q.shape[1]), alibi_slopes):
            Q = q[b, hq]  # [n, D]
            K = k[b, min(hq, k.shape[1]-1)]  # map to available KV head
            V = v[b, min(hq, v.shape[1]-1)]

            if DEBUG_TEST:
                print(f"\n[Batch {b}] Head {hq}  Q (shape {tuple(Q.shape)})")
                print(Q)
                print(f"[Batch {b}] Head {hq}  K (shape {tuple(K.shape)})")
                print(K)
                print(f"\n[Batch {b}] Head {hq}  V (shape {tuple(V.shape)})")
                print(V)

            m_val = m.item()
            sigma = 1.0 / m_val
            scores = K @ Q.T                       # [n, n]

            D = Q.shape[1]
            scores = scores/math.sqrt(D)

            for s in s_values:
                    alibi_matrix = _get_alibi_matrix(n, sigma, device=device)
                    hashes = sample_hashes(n=n, s=s, sigma=sigma, device=device)
                    average_matrix = generate_block_diagonal_masks_from_hashes(hashes)
                    print(f"\n[layer {layer_idx}] [head {hq}] m_val={m_val:.6g}  sigma={sigma:.6g}  n={n} s={s}")
                    
                    # Compute spectral norm (largest singular value)
                    diff_norm = torch.linalg.norm(average_matrix - alibi_matrix, ord=2).item()
                    diff_max_norm = (average_matrix - alibi_matrix).abs().max().item()

                    base_norm = torch.linalg.norm(alibi_matrix, ord=2).item()
                    epsilon = diff_norm
                    epsilon_emp = diff_norm / (base_norm + 1e-12)

                    print(f"  s={s:6d}  ‖L′−L‖₂={diff_norm:.6e}  max|L′−L|={diff_max_norm:.6e}  ‖L′−L‖₂/‖L‖₂={epsilon_emp:.6e}")

                    s_good = int(math.ceil((1 / (epsilon ** 2)) * (sigma ** 2) * logn))

                    print("Expected S value to give good results = ", s_good)
                    print("Expectation bound =", bound_expectation(sigma, n, s))
                    print("Tail bound =", bound_tail(sigma, n, s, epsilon))
                    if DEBUG_TEST:
                        print("average_matrix (shape={}):\n{}".format(tuple(getattr(average_matrix, "shape", [])),
                                                                      average_matrix))
                        print("alibi_matrix (shape={}):\n{}".format(tuple(getattr(alibi_matrix, "shape", [])), alibi_matrix))


                    scores = scores.to(device, dtype=torch.float32)
                    alibi_matrix = alibi_matrix.to(device, dtype=torch.float32)
                    average_matrix = average_matrix.to(device, dtype=torch.float32)
                    V = V.to(device, dtype=torch.float32)

                    print("scores:", scores.device, scores.dtype, scores.shape)
                    print("alibi_matrix:", alibi_matrix.device, alibi_matrix.dtype, alibi_matrix.shape)
                    print("average_matrix:", average_matrix.device, average_matrix.dtype, average_matrix.shape)
                    print("V:", V.device, V.dtype, V.shape)

                    E = torch.exp(scores)  # [n, n]
                    out1 = (E * alibi_matrix) @ V  # [n, D]
                    out2 = (E * average_matrix) @ V  # [n, D]
                    diff_norm_exp = torch.norm(out1 - out2).item()
                    diff_max_exp = (out1 - out2).abs().max().item()
                    print(f"‖exp(KQ)∘L·V - exp(KQ)∘L′·V‖ = {diff_norm_exp:.6f}")
                    print(f"max|exp(KQ)∘L·V - exp(KQ)∘L′·V| = {diff_max_exp:.6f}")

                    eps = 1e-20  # avoid log(0)
                    logL = (alibi_matrix.clamp_min(eps)).log()  # [n, n]
                    logLp = (average_matrix.clamp_min(eps)).log()

                    attn1 = torch.softmax(scores + logL, dim=-1)  # [n, n]
                    attn2 = torch.softmax(scores + logLp, dim=-1)

                    out1 = attn1 @ V
                    out2 = attn2 @ V
                    diff_norm_sm = torch.norm(out1 - out2).item()
                    print(f"‖softmax(KQ+log L)·V - softmax(KQ+log L′)·V‖ = {diff_norm_sm:.6f}")

                    # Debug stats on the attention distributions
                    print("attn1 stats: min={:.4g}, max={:.4g}, mean={:.4g}".format(
                        attn1.min().item(), attn1.max().item(), attn1.mean().item()))
                    print("attn2 stats: min={:.4g}, max={:.4g}, mean={:.4g}".format(
                        attn2.min().item(), attn2.max().item(), attn2.mean().item()))

                    # Norms of the output matrices
                    o1 = attn1 @ V
                    o2 = attn2 @ V
                    abs_norm = torch.linalg.norm(o1 - o2).item()
                    rel_norm = abs_norm / (0.5 * (torch.linalg.norm(o1).item() + torch.linalg.norm(o2).item()) + 1e-12)
                    print(f"Abs diff: {abs_norm:.6f}   Rel diff: {rel_norm:.3%}")

                    # Per-token (row-wise) relative diffs (median is robust)
                    row_abs = torch.linalg.norm(o1 - o2, dim=1)
                    row_ref = 0.5 * (torch.linalg.norm(o1, dim=1) + torch.linalg.norm(o2, dim=1)) + 1e-12
                    row_rel = (row_abs / row_ref).cpu()
                    print(f"Row-wise rel diff: median={row_rel.median().item():.3%}, "
                          f"p90={row_rel.quantile(0.9).item():.3%}, p99={row_rel.quantile(0.99).item():.3%}")

                    # Cosine similarity of the flattened outputs
                    cos = torch.nn.functional.cosine_similarity(o1.flatten(), o2.flatten(), dim=0).item()
                    print(f"Cosine(o1, o2) = {cos:.6f}")

from __future__ import annotations

import torch


def sample_hashes(
    n: int,
    s: int,
    sigma: float,
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
    seed: int | None = None,
    sample: bool = True,
) -> torch.Tensor:
    """
    Vectorised but memory-safe sampling of the bucket assignments used in the LSH masks.
    Processes in smaller batches to avoid (s,n) blowing up on GPU.

    Args:
        n (int): sequence length
        s (int): number of sampled block-diagonal matrices
        sigma (float): inverse slope 1/m in ALiBi bias
        device (torch.device, optional): where to put results
        batch_size (int): number of rows to process at once
        seed (int, optional): random seed for reproducible sampling
        sample (bool, optional, default True): if True, draw b ~ Gamma(2, sigma) and
            xi ~ Uniform(0, b) randomly. If False, use distribution means:
            b = 2*sigma (E[Gamma(2,sigma)]) and xi = 0.

    Returns:
        LongTensor of shape (s, n)
    """
    if device is None:
        device = torch.device("cpu")

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    idx = torch.arange(n, device=device).view(1, -1)  # (1, n)

    out = []
    for start in range(0, s, batch_size):
        c = min(batch_size, s - start)

        if sample:
            # Gamma(k=2, theta=sigma) via sum of 2 exponentials:
            # if U ~ Uniform(0,1), then -sigma * log(U) ~ Exponential(scale=sigma)
            u1 = torch.rand((c, 1), device=device, generator=gen).clamp_min(1e-12)
            u2 = torch.rand((c, 1), device=device, generator=gen).clamp_min(1e-12)
            b = -sigma * (torch.log(u1) + torch.log(u2))              # (c, 1)
            xi = torch.rand((c, 1), device=device, generator=gen) * b  # (c, 1)
        else:
            # Use distribution means: E[Gamma(2, sigma)] = 2*sigma, xi = 0.
            b = torch.full((c, 1), 2.0 * sigma, dtype=torch.float32, device=device)  # (c, 1)
            xi = torch.zeros((c, 1), dtype=torch.float32, device=device)             # (c, 1)

        h = torch.floor_divide(idx - xi, b).long()  # (c, n)
        out.append(h)

    return torch.cat(out, dim=0)  # (s, n)


def generate_block_diagonal_masks_from_hashes(hashes: torch.Tensor, *, chunk_size: int | None = None) -> torch.Tensor:
    """
    Compute average block-diagonal mask from sampled hash assignments without materializing (s, n, n).

    Args:
        hashes: LongTensor of shape (s, n), bucket IDs per token for each of s hash tables.
        chunk_size: optional number of hash tables to process at once (for a speed/memory tradeoff).
                    If None, processes one table at a time (lowest memory).

    Returns:
        average_matrix: FloatTensor of shape (n, n), average of s block-diagonal binary masks.
    """
    device = hashes.device
    dtype = torch.float32
    s, n = hashes.shape

    avg = torch.zeros((n, n), device=device, dtype=dtype)

    if chunk_size is None or chunk_size <= 0:
        # Minimal memory: one hash table at a time
        for i in range(s):
            h = hashes[i]                          # (n,)
            mask = (h[None, :] == h[:, None])      # (n, n) boolean
            avg += mask.to(dtype)
    else:
        # Process in chunks of size `chunk_size`: (c, n) -> (c, n, n) then reduce
        for start in range(0, s, chunk_size):
            end = min(start + chunk_size, s)
            h = hashes[start:end]                  # (c, n)
            # (c, n, 1) == (c, 1, n) -> (c, n, n), then mean over c
            matches_c = (h.unsqueeze(2) == h.unsqueeze(1)).to(dtype)  # (c, n, n)
            avg += matches_c.mean(dim=0) * (end - start)              # weight by c

    avg /= s
    return avg

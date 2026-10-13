"""Checks for the claims speculative decoding rests on. CPU, about 30 seconds, no downloads.

    python check.py
"""

import torch

from llama import LM, Config, cache_len, rope, truncate
from speculative import accept_or_resample, autoregressive, speculative

torch.manual_seed(0)


def tiny(seed: int, layers: int = 2) -> LM:
    torch.manual_seed(seed)
    return LM(Config(vocab=50, d_model=48, d_ff=96, n_layers=layers, n_heads=6, n_kv_heads=2, qkv_bias=True)).eval()


def check_output_distribution_is_the_targets() -> None:
    """Theorem 3.5: draw from q, accept with min(1, p/q), else resample from norm(max(0, p - q)). The tokens
    that come out are distributed exactly as p, whatever q is."""
    gen = torch.Generator().manual_seed(0)
    p = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    q = torch.tensor([0.05, 0.10, 0.30, 0.30, 0.15, 0.10])  # a deliberately bad draft
    n, counts = 60_000, torch.zeros(6)
    for _ in range(n):
        x = torch.multinomial(q, 1, generator=gen).item()
        r = accept_or_resample(p, q, x, gen)
        counts[x if r is None else r] += 1
    tv = 0.5 * (counts / n - p).abs().sum().item()
    assert tv < 0.01, f"total variation {tv:.4f} from the target distribution"


def check_kv_cache_and_rollback() -> None:
    m = tiny(1)
    x = torch.randint(0, 50, (1, 12))
    full = m(x)
    cache = m.new_cache()
    steps = torch.cat([m(x[:, :5], cache)] + [m(x[:, i : i + 1], cache) for i in range(5, 12)], 1)
    assert torch.allclose(steps, full, atol=1e-5), "KV cache changes the output"
    truncate(cache, 7)  # forget tokens 7..11, as after a rejection, then feed different ones
    y = torch.randint(0, 50, (1, 3))
    after = m(y, cache)
    assert cache_len(cache) == 10
    assert torch.allclose(after, m(torch.cat([x[:, :7], y], 1))[:, 7:], atol=1e-5), "rollback left stale keys"


def check_rope_is_relative() -> None:
    """After RoPE, a query-key dot product depends only on the distance between the positions."""
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    dot = lambda i, j: (rope(q, torch.tensor([i]), 1e4) * rope(k, torch.tensor([j]), 1e4)).sum().item()
    assert abs(dot(3, 1) - dot(10, 8)) < 1e-4 and abs(dot(3, 1) - dot(3, 2)) > 1e-4


def copy_model(seed: int, steps: int) -> LM:
    """Trained to continue a period-3 pattern (x_t = x_{t-3}): the next token depends on context three back,
    so a cache holding the wrong tokens or positions changes the output. Fewer steps = a worse model."""
    m = tiny(seed).train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randint(0, 50, (32, 3), generator=gen).repeat(1, 8)
        loss = torch.nn.functional.cross_entropy(m(x[:, :-1])[:, 2:].flatten(0, 1), x[:, 3:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def check_greedy_speculative_equals_target_greedy() -> None:
    """With temperature 0 the output must be token-for-token the target's own greedy output, for any draft."""
    target, draft = copy_model(1, 300), copy_model(2, 100)
    rates = []
    for seed in range(3):
        prompt = torch.randint(0, 50, (3,), generator=torch.Generator().manual_seed(seed)).tolist()
        want, _ = autoregressive(target, prompt, 20)
        for gamma in (1, 3, 5):
            got, stats = speculative(target, draft, prompt, 20, gamma=gamma)
            assert got == want, (seed, gamma, got, want)
            rates.append(stats.acceptance)
    assert max(rates) > 0 and min(rates) < 1, f"both accept and reject paths must run, acceptance {rates}"


def check_perfect_draft_is_always_accepted() -> None:
    """If the draft IS the target, every guess is accepted (p/q = 1): gamma + 1 tokens per target pass."""
    m = tiny(1)
    _, stats = speculative(m, m, [1, 2, 3], 20, gamma=4, temperature=1.0)
    assert stats.acceptance == 1.0 and set(stats.per_pass) == {5}, stats


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("check_"):
            fn()
            print(f"ok  {name[6:]}")

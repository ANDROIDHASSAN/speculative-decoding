"""Speculative sampling (Leviathan, Kalman & Matias 2023, arXiv 2211.17192; also Chen et al. 2023).

A large target model p is slow because it produces one token per forward pass, and that pass is
memory-bound: reading the weights costs the same for 1 token as for 5. So let a small draft model q guess
gamma tokens, and let p score all of them in ONE pass. Algorithm 1 of the paper:

    1. draft:  x_1..x_g ~ q, one at a time (cheap)
    2. verify: p(. | prefix), p(. | prefix, x_1), ..., p(. | prefix, x_1..x_g) in one target pass
    3. accept x_i with probability min(1, p(x_i) / q(x_i)), left to right; at the first rejection, sample
       a replacement from norm(max(0, p - q)) and stop
    4. if all g were accepted, sample one more token from the last target distribution

Each target pass yields between 1 and g + 1 tokens, and the output distribution is EXACTLY p's (Theorem 3.5):
accepting with min(1, p/q) and resampling the leftover mass reproduces p for every token. check.py measures
that empirically. With temperature 0 the same loop reduces to "keep draft tokens while they equal the target's
argmax", and the output is identical, token for token, to greedy decoding of the target alone.
"""

import time
from dataclasses import dataclass, field

import torch

from llama import LM, cache_len, truncate


@dataclass
class Stats:
    target_passes: int = 0
    drafted: int = 0
    accepted: int = 0
    seconds: float = 0.0
    per_pass: list[int] = field(default_factory=list)  # tokens produced by each target pass

    @property
    def acceptance(self) -> float:
        return self.accepted / max(self.drafted, 1)


def probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, -1)


def accept_or_resample(p: torch.Tensor, q: torch.Tensor, token: int, gen: torch.Generator) -> int | None:
    """One step of Algorithm 1. Returns None if `token` (drawn from q) is accepted, else the replacement
    drawn from norm(max(0, p - q))."""
    if torch.rand((), generator=gen, device=p.device) < torch.clamp(p[token] / q[token], max=1.0):
        return None
    residual = torch.clamp(p - q, min=0.0)
    return torch.multinomial(residual / residual.sum(), 1, generator=gen).item()


def feed(model: LM, cache: list, out: list[int], device) -> torch.Tensor:
    """Run the model on every token of `out` not yet in its cache; logits for the last one."""
    return model(torch.tensor([out[cache_len(cache) :]], device=device), cache)[0, -1]


@torch.no_grad()
def autoregressive(model: LM, prompt: list[int], max_new: int, temperature: float = 0.0, eos: int | None = None,
                   seed: int = 0) -> tuple[list[int], Stats]:
    """The baseline: one target pass per token."""
    device, gen = next(model.parameters()).device, torch.Generator(next(model.parameters()).device).manual_seed(seed)
    cache, out, stats, t0 = model.new_cache(), list(prompt), Stats(), time.perf_counter()
    while len(out) - len(prompt) < max_new:
        logits = feed(model, cache, out, device)
        stats.target_passes += 1
        tok = logits.argmax().item() if temperature == 0 else torch.multinomial(probs(logits, temperature), 1, generator=gen).item()
        out.append(tok)
        stats.per_pass.append(1)
        if tok == eos:
            break
    stats.seconds = time.perf_counter() - t0
    return out[len(prompt) :], stats


@torch.no_grad()
def speculative(target: LM, draft: LM, prompt: list[int], max_new: int, gamma: int = 4, temperature: float = 0.0,
                eos: int | None = None, seed: int = 0) -> tuple[list[int], Stats]:
    device = next(target.parameters()).device
    gen = torch.Generator(device).manual_seed(seed)
    t_cache, d_cache = target.new_cache(), draft.new_cache()
    out, stats, t0 = list(prompt), Stats(), time.perf_counter()
    while len(out) - len(prompt) < max_new:
        # 1. draft gamma tokens, keeping each q distribution
        guesses, qs = [], []
        for _ in range(gamma):
            logits = feed(draft, d_cache, out + guesses, device)
            if temperature == 0:
                tok = logits.argmax().item()
            else:
                q = probs(logits, temperature)
                tok = torch.multinomial(q, 1, generator=gen).item()
                qs.append(q)
            guesses.append(tok)
        # 2. one target pass over the pending token and all guesses: gamma + 1 distributions
        start = cache_len(t_cache)
        logits = target(torch.tensor([(out + guesses)[start:]], device=device), t_cache)[0, -(gamma + 1) :]
        stats.target_passes += 1
        stats.drafted += gamma
        # 3. accept left to right; the first rejection is replaced and ends the pass
        new, n_accepted = [], 0
        for i, tok in enumerate(guesses):
            if temperature == 0:
                best = logits[i].argmax().item()
                replacement = None if best == tok else best
            else:
                replacement = accept_or_resample(probs(logits[i], temperature), qs[i], tok, gen)
            if replacement is not None:
                new.append(replacement)
                break
            new.append(tok)
            n_accepted += 1
            if tok == eos:
                break
        else:  # 4. every guess accepted: the target pass already gives the next token for free
            last = logits[gamma]
            new.append(last.argmax().item() if temperature == 0 else torch.multinomial(probs(last, temperature), 1, generator=gen).item())
        stats.accepted += n_accepted
        out += new
        stats.per_pass.append(len(new))
        # 5. both caches must hold exactly the tokens before the new pending one: drop rejected guesses
        truncate(t_cache, len(out) - 1)
        truncate(d_cache, min(cache_len(d_cache), len(out) - 1))
        if eos in new:
            break
    produced = out[len(prompt) :]
    if eos in produced:
        produced = produced[: produced.index(eos) + 1]
    stats.seconds = time.perf_counter() - t0
    return produced[:max_new], stats

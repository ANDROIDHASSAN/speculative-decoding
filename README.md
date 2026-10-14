# Speculative decoding from scratch — a small model drafts, a big one verifies

A from-scratch implementation of **speculative sampling** ([Leviathan, Kalman & Matias, 2023](https://arxiv.org/abs/2211.17192);
[Chen et al., 2023](https://arxiv.org/abs/2302.01318)) on top of a Llama-family transformer written by hand,
which loads real Hugging Face checkpoints (Qwen2.5 0.5B as the draft, Qwen2.5 1.5B as the target).

The idea: a large model's forward pass is memory-bound, so scoring 5 tokens costs about as much as scoring 1.
A small draft model guesses γ tokens; the target checks all of them in **one** pass and keeps the prefix it
agrees with. The accept/reject rule makes the output distribution **exactly** the target's, so the speed-up
is free of quality loss.

## What's here

| File | What |
|---|---|
| [llama.py](llama.py) | Llama / Qwen2 decoder: RMSNorm, rotary positions (HF layout), grouped-query attention, SwiGLU, a KV cache that can be truncated, and a safetensors loader for HF checkpoints |
| [speculative.py](speculative.py) | The paper's Algorithm 1: draft γ tokens, verify in one target pass, accept with min(1, p/q), resample from norm(max(0, p − q)), bonus token when all are accepted; the greedy special case; the target-only baseline |
| [bench.py](bench.py) | Real models: tokens/s, speed-up, acceptance rate α, tokens per target pass vs the paper's prediction (1 − α^(γ+1)) / (1 − α) |
| [check.py](check.py) | The claims below, on CPU, no downloads |

Loading `Qwen/Qwen2.5-0.5B-Instruct` into `llama.py` gives 494M parameters (the official count) and answers
"What is the capital of France?" with "The capital of France is Paris."

## Results: Qwen2.5 1.5B target, 0.5B draft, Apple M1 (MPS, bf16)

`python bench.py` (greedy) and `python bench.py --temperature 0.7`, batch size 1:

| γ | greedy tok/s | speed-up | α | tokens / target pass (measured) | predicted from α |
|---|---|---|---|---|---|
| target alone | 23.8 | 1.00x | | 1.00 | |
| 2 | 11.9 | 0.50x | 0.72 | 2.42 | 2.23 |
| 4 | 19.0 | 0.80x | 0.61 | 3.37 | 2.33 |
| 6 | 16.9 | 0.71x | 0.50 | 3.95 | 1.98 |

| γ | T = 0.7 tok/s | speed-up | α | tokens / target pass (measured) | predicted from α |
|---|---|---|---|---|---|
| target alone | 24.6 | 1.00x | | 1.00 | |
| 2 | 23.6 | 0.96x | 0.81 | 2.57 | 2.46 |
| 4 | 18.2 | 0.74x | 0.59 | 3.31 | 2.25 |
| 6 | 15.4 | 0.62x | 0.46 | 3.66 | 1.83 |

The algorithm does what the paper says, in target passes: each pass of the 1.5B model yields 2.4–4.0 tokens
instead of 1. The wall clock does not follow on this machine: every row is slower than the target alone. Two
likely reasons, not separately measured: the draft is a third of the target's size (the paper's gains assume a
much cheaper draft), and at this model size an MPS forward pass is dominated by per-step overhead rather than
memory bandwidth, so verifying γ tokens is not as free as on a large GPU model. The machine was also under
memory pressure during the run (19–22 GB of swap in use).

In greedy mode the output leaves the target-alone output after 12 tokens: bf16 rounding differs between a
1-token and a (γ + 1)-token forward pass, so near-tied logits can flip. Run in fp32 on CPU, the same two models give
identical greedy outputs, and `greedy_speculative_equals_target_greedy` in `check.py` checks the exact match
in fp32.

## Correctness checks

`python check.py`:

| Check | What it proves |
|---|---|
| `output_distribution_is_the_targets` | Theorem 3.5: with a deliberately bad draft q, tokens produced by accept-or-resample follow the target p (total variation < 0.01 over 60k draws). Resampling from p instead of max(0, p − q) gives 0.18 and fails |
| `greedy_speculative_equals_target_greedy` | At temperature 0 the output equals the target's own greedy output token for token, for γ = 1, 3, 5, with a partly-right draft (both accept and reject paths run). Models are trained on a task whose next token depends on context, so a stale cache would change the output |
| `kv_cache_and_rollback` | Incremental decoding equals the full pass; after truncating the cache to drop rejected tokens, new tokens see exactly the kept prefix |
| `rope_is_relative` | After rotary embedding, a query-key score depends only on the distance between positions |
| `perfect_draft_is_always_accepted` | If draft = target, p/q = 1: every guess is accepted, γ + 1 tokens per target pass |

Breaking the residual distribution, or skipping either cache rollback, makes a check fail.

## Run it

```bash
pip install -r requirements.txt
python check.py
python bench.py                       # greedy; reports any token where the output leaves the target alone (bf16 rounding)
python bench.py --temperature 0.7     # sampling
```

## Notes

- Both models must share a tokenizer (Qwen2.5 0.5B / 1.5B do: 151,936 tokens).
- The draft's cache is also rolled back, so the draft never conditions on a token the target rejected.
- Batch size 1, as in the paper's latency setting.

## References

- Leviathan, Kalman & Matias, 2023. *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al., 2023. *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Su et al., 2021. *RoFormer: Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Ainslie et al., 2023. *GQA: Grouped-Query Attention.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Qwen Team, 2024. *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

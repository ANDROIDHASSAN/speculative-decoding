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

> **Status:** algorithm and model code complete and checked. The real-model benchmark (`bench.py`) runs next;
> its table will be added here.

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

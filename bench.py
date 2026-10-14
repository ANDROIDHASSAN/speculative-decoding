"""Speed and acceptance on real models: Qwen2.5-1.5B-Instruct (target) with Qwen2.5-0.5B-Instruct (draft),
same tokenizer, both loaded into llama.py from their Hugging Face checkpoints.

    python bench.py                       # greedy: also reports where (if anywhere) the output leaves the target-only output
    python bench.py --temperature 0.7

Prints, per gamma: wall-clock tokens/s, speedup over the target alone, draft acceptance rate alpha, tokens
per target pass, and the paper's prediction of tokens per pass from alpha: (1 - alpha^(gamma+1)) / (1 - alpha).
"""

import argparse

import torch

from llama import load_pretrained, n_params, tokenizer
from speculative import autoregressive, speculative

PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a short Python function that checks whether a number is prime.",
    "भारत की राजधानी के बारे में दो वाक्य लिखिए।",
    "List five tips for a job interview.",
]


def chat(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gammas", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    target = load_pretrained(args.target, dtype, args.device)
    draft = load_pretrained(args.draft, dtype, args.device)
    tok = tokenizer(args.target)
    eos = tok.token_to_id("<|im_end|>")
    print(f"target {n_params(target) / 1e9:.2f}B, draft {n_params(draft) / 1e9:.2f}B, {args.device}, {dtype}, "
          f"temperature {args.temperature}")
    prompts = [tok.encode(chat(p)).ids for p in PROMPTS]
    autoregressive(target, prompts[0], 8)  # warm-up: first-call kernel compilation is not decoding speed
    speculative(target, draft, prompts[0], 8)

    base_tokens = base_time = 0
    baseline = []
    for p in prompts:
        out, st = autoregressive(target, p, args.max_new, args.temperature, eos)
        baseline.append(out)
        base_tokens, base_time = base_tokens + len(out), base_time + st.seconds
    base_rate = base_tokens / base_time
    print(f"\n| gamma | tokens/s | speedup | acceptance alpha | tokens per target pass | predicted from alpha |")
    print(f"|---|---|---|---|---|---|")
    print(f"| target alone | {base_rate:.1f} | 1.00x | | 1.00 | |")
    for gamma in args.gammas:
        tokens = seconds = drafted = accepted = passes = 0
        for p, want in zip(prompts, baseline):
            out, st = speculative(target, draft, p, args.max_new, gamma, args.temperature, eos)
            if args.temperature == 0 and out != want:
                # Exact in float32 (check.py); in bfloat16 the target scoring gamma+1 tokens in one pass rounds
                # differently from scoring them one at a time, so a near-tie between two tokens can flip.
                first = next((i for i, (a, b) in enumerate(zip(out, want)) if a != b), min(len(out), len(want)))
                print(f"  gamma {gamma}: output differs from the target alone from token {first} of {len(want)}")
            tokens, seconds = tokens + len(out), seconds + st.seconds
            drafted, accepted, passes = drafted + st.drafted, accepted + st.accepted, passes + st.target_passes
        alpha = accepted / drafted
        predicted = (1 - alpha ** (gamma + 1)) / (1 - alpha) if alpha < 1 else gamma + 1
        print(f"| {gamma} | {tokens / seconds:.1f} | {tokens / seconds / base_rate:.2f}x | {alpha:.2f} | "
              f"{tokens / passes:.2f} | {predicted:.2f} |")
    print(f"\nsample ({PROMPTS[2]}): {tok.decode(baseline[2])[:200]}")


if __name__ == "__main__":
    main()

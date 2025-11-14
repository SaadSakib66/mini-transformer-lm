# src/inference.py

import sys
import argparse
import torch
import torch.nn.functional as F

from model import TinyTransformerLM


def load_model_and_tokenizer(checkpoint_path: str = "../tiny_transformer.pt", device: str = None):
    """
    Load trained TinyTransformerLM and tokenizer info from checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab_size = ckpt["vocab_size"]
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    config = ckpt["config"]

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[INFO] Model loaded (vocab_size={vocab_size}, d_model={config['d_model']})")
    return model, stoi, itos, device, config


def encode_prompt(prompt: str, stoi: dict):
    """
    Convert prompt string -> list of token ids.
    Unknown chars are skipped with a warning.
    """
    ids = []
    for ch in prompt:
        if ch not in stoi:
            print(f"[WARN] Character {repr(ch)} not in vocabulary. Skipping.")
            continue
        ids.append(stoi[ch])
    if not ids:
        raise ValueError("Prompt has no valid characters from the vocabulary.")
    return ids


def decode_ids(ids, itos: dict):
    """
    Convert list of token ids -> string.
    """
    return "".join(itos[i] for i in ids)


@torch.no_grad()
def sample_with_temperature(
    model: TinyTransformerLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    device: str,
    max_seq_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    """
    Custom autoregressive generation with:
    - temperature
    - top-k sampling

    Args:
        model: trained TinyTransformerLM
        idx: (1, current_seq_len) starting token ids
        max_new_tokens: how many new tokens to generate
        device: 'cpu' or 'cuda'
        max_seq_len: model's maximum sequence length
        temperature: >1.0 = more random, <1.0 = more greedy, 1.0 = default
        top_k: if not None, only keep top_k logits before softmax
    """
    model.eval()
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        # crop context to last max_seq_len tokens
        idx_cond = idx[:, -max_seq_len:]

        # forward pass
        logits = model(idx_cond)  # (1, seq_len, vocab_size)

        # take last time step
        logits = logits[:, -1, :]  # (1, vocab_size)

        # apply temperature
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = logits / temperature

        # optional top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, k=top_k, dim=-1)
            min_keep = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_keep, torch.full_like(logits, float('-inf')), logits)

        # softmax -> probabilities
        probs = F.softmax(logits, dim=-1)  # (1, vocab_size)

        # sample from distribution
        next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

        # append to sequence
        idx = torch.cat([idx, next_token], dim=1)  # (1, seq_len+1)

    return idx


def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    checkpoint_path: str = "../tiny_transformer.pt",
    temperature: float = 1.0,
    top_k: int | None = None,
    num_samples: int = 1,
):
    """
    High-level helper: load model, encode prompt, and generate multiple samples.
    """
    model, stoi, itos, device, config = load_model_and_tokenizer(checkpoint_path, device=None)

    # encode prompt
    start_ids = encode_prompt(prompt, stoi)
    start_idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    outputs = []
    for i in range(num_samples):
        print(f"\n[INFO] Generating sample {i+1}/{num_samples} ...")
        out_ids = sample_with_temperature(
            model=model,
            idx=start_idx.clone(),  # clone so each sample starts from same prompt
            max_new_tokens=max_new_tokens,
            device=device,
            max_seq_len=config["max_seq_len"],
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()

        text = decode_ids(out_ids, itos)
        outputs.append(text)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny Transformer LM Inference Script (with temperature & top-k sampling)"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=None,
        help="Prompt text to start generation. If omitted, you will be asked interactively.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate (default: 100).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature >0 (default: 1.0). Higher = more random.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="If >0, use top-k sampling with this k (default: 0 = disabled).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="How many samples to generate from the same prompt (default: 1).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../tiny_transformer.pt",
        help="Path to model checkpoint (default: ../tiny_transformer.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If no prompt passed as CLI arg, ask interactively
    if args.prompt is None:
        prompt = input("Enter prompt text: ")
    else:
        prompt = args.prompt

    top_k = args.top_k if args.top_k > 0 else None

    print(f"\n[CONFIG]")
    print(f"  Prompt         : {repr(prompt)}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Top-k          : {top_k}")
    print(f"  Num samples    : {args.num_samples}")
    print(f"  Checkpoint     : {args.ckpt}")

    outputs = generate_text(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        checkpoint_path=args.ckpt,
        temperature=args.temperature,
        top_k=top_k,
        num_samples=args.num_samples,
    )

    print("\n================= GENERATED TEXT =================\n")
    for i, text in enumerate(outputs, start=1):
        print(f"----- Sample {i} -----")
        print(text)
        print()


if __name__ == "__main__":
    main()

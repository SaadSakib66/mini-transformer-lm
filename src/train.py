# src/train.py

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt  # NEW: for plotting

from data import load_data
from model import TinyTransformerLM


def train(
    corpus_path: str = "../data/corpus.txt",
    block_size: int = 32,
    batch_size: int = 32,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 256,
    max_seq_len: int = 64,
    dropout: float = 0.1,
    lr: float = 3e-4,
    epochs: int = 50,
    device: str = None,
    save_path: str = "../tiny_transformer.pt",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Load data (NOTE: train_split=0.8 so we get some validation data)
    train_loader, val_loader, tokenizer = load_data(
        corpus_path=corpus_path,
        block_size=block_size,
        batch_size=batch_size,
        train_split=0.8,  # changed from default 0.9
    )

    # 2. Build model
    model = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # 4. For logging losses
    train_losses = []
    val_losses = []

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for x, y in pbar:
            x = x.to(device)   # (batch, seq_len)
            y = y.to(device)   # (batch, seq_len)

            optimizer.zero_grad()

            logits = model(x)  # (batch, seq_len, vocab_size)

            # CrossEntropyLoss expects (N, C) and target (N)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), y.view(B * T))

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_sum / num_batches

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                B, T, C = logits.shape
                vloss = criterion(logits.view(B * T, C), y.view(B * T))
                val_loss_sum += vloss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(1, val_batches)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    # 6. Save model & tokenizer info
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "config": {
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "block_size": block_size,
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, save_path)
    print(f"Model saved to {save_path}")

    # 7. Plot loss curves code 
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save plot next to model file
    loss_plot_path = os.path.join(os.path.dirname(save_path), "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {loss_plot_path}")

    # 8. Quick demo generation
    demo_prompt = "amar"
    print("\n=== Demo generation ===")
    print("Prompt:", demo_prompt)

    # encode prompt
    start_ids = [tokenizer.stoi[c] for c in demo_prompt if c in tokenizer.stoi]
    if not start_ids:
        print("Prompt characters not in vocab, skipping demo.")
        return

    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    gen_ids = model.generate(idx, max_new_tokens=100)[0].tolist()
    generated_text = tokenizer.decode(gen_ids)
    print("Generated:", generated_text)


if __name__ == "__main__":
    train()


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (batch, seq_len)
        return self.embedding(x)  # (batch, seq_len, d_model)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding: position index -> vector
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len)
        b, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return self.embedding(positions)  # (1, seq_len, d_model)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        b, seq_len, _ = x.size()

        # 1. Linear projections
        Q = self.W_q(x)  # (b, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into heads
        # (b, seq_len, num_heads, d_head) -> (b, num_heads, seq_len, d_head)
        Q = Q.view(b, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(b, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(b, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # 3. Scaled dot-product attention
        # scores: (b, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            # mask expected shape: (1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (b, num_heads, seq_len, d_head)

        # 4. Concatenate heads
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)

        # 5. Final linear
        out = self.W_o(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + layernorm
        attn_out = self.attn(x, mask)
        x = self.ln1(x + self.dropout(attn_out))

        # Feed-forward + residual + layernorm
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class TinyTransformerLM(nn.Module):
    """
    Tiny decoder-only Transformer Language Model
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _generate_causal_mask(self, seq_len: int, device):
        # lower-triangular matrix (1: allowed, 0: masked)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, seq_len, seq_len)

    def forward(self, x):
        # x: (batch, seq_len)
        b, seq_len = x.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} > max_seq_len={self.max_seq_len}")

        token_emb = self.token_emb(x)                # (b, seq_len, d_model)
        pos_emb = self.pos_emb(x)                   # (1, seq_len, d_model)
        h = token_emb + pos_emb                     # broadcasting
        h = self.dropout(h)

        mask = self._generate_causal_mask(seq_len, x.device)

        for block in self.blocks:
            h = block(h, mask=mask)

        h = self.ln_final(h)
        logits = self.head(h)                       # (b, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """
        Autoregressive generation:
        idx: (1, current_seq_len) start tokens
        """
        for _ in range(max_new_tokens):
            # crop to max_seq_len
            idx_cond = idx[:, -self.max_seq_len:]

            # forward
            logits = self(idx_cond)  # (1, seq_len, vocab_size)

            # last time step
            logits_last = logits[:, -1, :]  # (1, vocab_size)

            # softmax -> probabilities
            probs = F.softmax(logits_last, dim=-1)

            # sample next token (or use argmax)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # append
            idx = torch.cat([idx, next_token], dim=1)
        return idx

"""Small models for grokking: a 1-layer transformer (Nanda et al. style) and an MLP."""
import torch
import torch.nn as nn


class Transformer(nn.Module):
    """1-layer, no-LayerNorm transformer in the style of Nanda et al. (2023).

    Only the final ('=') position is read out, and with one layer nothing downstream
    depends on the other positions' outputs. So queries, the attention output and the
    MLP are computed only at the last position, while keys and values use all positions.
    This is mathematically identical to a causal 1-layer transformer read at the last
    token, and about 3x cheaper.
    """

    def __init__(self, vocab: int, n_out: int, d_model=128, n_heads=4, d_mlp=512, seq_len=4):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.embed = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.embed.weight, std=d_model**-0.5)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) / d_model**0.5)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d_model))
        self.unembed = nn.Linear(d_model, n_out, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos[:T]                                    # B T D
        k, v = self.kv(h).view(B, T, 2, self.n_heads, self.d_head).unbind(2)  # B T H d
        q = self.q(h[:, -1]).view(B, self.n_heads, self.d_head)            # B H d
        att = torch.einsum("bhd,bthd->bht", q, k) / self.d_head**0.5
        z = torch.einsum("bht,bthd->bhd", att.softmax(-1), v).reshape(B, -1)
        r = h[:, -1] + self.o(z)
        r = r + self.mlp(r)
        return self.unembed(r)


class MLP(nn.Module):
    """Embeds a and b, concatenates, and maps through a 1-hidden-layer MLP."""

    def __init__(self, vocab: int, n_out: int, d_embed=128, d_hidden=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_embed)
        self.net = nn.Sequential(
            nn.Linear(2 * d_embed, d_hidden), nn.ReLU(), nn.Linear(d_hidden, n_out)
        )

    def forward(self, x):
        e = self.embed(x[:, [0, 2]])  # just a and b
        return self.net(e.flatten(1))


def build_model(name: str, p: int, **kw):
    vocab, n_out = p + 2, p
    if name == "transformer":
        return Transformer(vocab, n_out, **kw)
    if name == "mlp":
        return MLP(vocab, n_out, **kw)
    raise ValueError(name)

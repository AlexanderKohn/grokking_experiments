"""Small models for grokking: a 1-layer transformer (Nanda et al. style) and an MLP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """1-hidden-layer ReLU MLP on (a, b). Only a and b are used; op and '=' are ignored.

    input modes:
      onehot: concat(onehot(a), onehot(b)) of size 2p -> hidden. This is the standard
              MLP-grokking setup (Gromov 2023; Liu et al. 2022). The first layer is
              equivalent to separate, untied embeddings for a and b that are summed.
      scalar: [a, b] / (p - 1) as two real numbers -> hidden. There is no symbolic
              structure, so the network must build the wrap-around itself, which is
              much harder.
      embed:  a shared learned embedding table for a and b, concatenated -> hidden.
    """

    def __init__(self, p: int, n_out: int, input_mode="onehot", d_hidden=512, d_embed=128):
        super().__init__()
        self.p, self.input_mode = p, input_mode
        if input_mode == "onehot":
            d_in = 2 * p
        elif input_mode == "scalar":
            d_in = 2
        elif input_mode == "embed":
            self.embed = nn.Embedding(p, d_embed)
            nn.init.normal_(self.embed.weight, std=d_embed**-0.5)
            d_in = 2 * d_embed
        else:
            raise ValueError(input_mode)
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, n_out))

    def forward(self, x):
        ab = x[:, [0, 2]]
        if self.input_mode == "onehot":
            h = F.one_hot(ab, self.p).float().flatten(1)
        elif self.input_mode == "scalar":
            h = ab.float() / (self.p - 1)
        else:
            h = self.embed(ab).flatten(1)
        return self.net(h)


def build_model(name: str, p: int, mlp_input: str = "onehot"):
    vocab, n_out = p + 2, p
    if name == "transformer":
        return Transformer(vocab, n_out)
    if name == "mlp":
        return MLP(p, n_out, input_mode=mlp_input)
    raise ValueError(name)

"""PyTorch trainer and sampler for the conditional MAF in npe.py (Intel Arc via XPU when present).

Same parameterisation as npe.log_prob, so exported weights reproduce the network's log density in
numpy (checked in the sandbox attestation).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .npe import S_MAX, made_masks


def device() -> torch.device:
    return torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")


class MADELayer(nn.Module):
    def __init__(self, d: int, k: int, hidden: int):
        super().__init__()
        m1, m2, m3 = (torch.tensor(m, dtype=torch.float32) for m in made_masks(d, hidden))
        self.register_buffer("m1", m1)
        self.register_buffer("m2", m2)
        self.register_buffer("m3", m3)
        self.W1 = nn.Parameter(torch.randn(hidden, d) / np.sqrt(d))
        self.V1 = nn.Parameter(torch.randn(hidden, k) / np.sqrt(k))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, hidden) / np.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(hidden))
        self.W3 = nn.Parameter(torch.randn(2 * d, hidden) * 1e-3)
        self.b3 = nn.Parameter(torch.zeros(2 * d))
        self.d = d

    def mu_s(self, x, c):
        h = torch.tanh(x @ (self.W1 * self.m1).T + c @ self.V1.T + self.b1)
        h = torch.tanh(h @ (self.W2 * self.m2).T + self.b2)
        out = h @ (self.W3 * self.m3).T + self.b3
        return out[:, : self.d], S_MAX * torch.tanh(out[:, self.d:] / S_MAX)

    def forward(self, x, c):
        mu, s = self.mu_s(x, c)
        return (x - mu) * torch.exp(-s), s

    def inverse(self, u, c):
        x = torch.zeros_like(u)
        for i in range(self.d):  # autoregressive: x_i needs x_<i
            mu, s = self.mu_s(x, c)
            x[:, i] = u[:, i] * torch.exp(s[:, i]) + mu[:, i]
        return x


class MAF(nn.Module):
    def __init__(self, d: int, k: int, hidden: int = 64, n_layers: int = 5):
        super().__init__()
        self.layers = nn.ModuleList([MADELayer(d, k, hidden) for _ in range(n_layers)])
        self.d, self.k, self.hidden = d, k, hidden

    def log_prob(self, x, c):
        total = torch.zeros(x.shape[0], device=x.device)
        u = x
        for n, layer in enumerate(self.layers):
            u, s = layer(u, c)
            total = total - s.sum(-1)
            if n < len(self.layers) - 1:
                u = torch.flip(u, dims=[-1])
        return total - 0.5 * (u**2).sum(-1) - 0.5 * self.d * np.log(2 * np.pi)

    @torch.no_grad()
    def sample(self, c, n: int):
        """n samples per context row: (rows, n, d)."""
        cc = c.repeat_interleave(n, dim=0)
        u = torch.randn(cc.shape[0], self.d, device=c.device)
        for idx in range(len(self.layers) - 1, -1, -1):
            if idx < len(self.layers) - 1:
                u = torch.flip(u, dims=[-1])
            u = self.layers[idx].inverse(u, cc)
        return u.reshape(c.shape[0], n, self.d)

    def export(self) -> list[dict]:
        return [{k: getattr(l, k).detach().cpu().double().numpy() for k in ("W1", "V1", "b1", "W2", "b2", "W3", "b3")}
                for l in self.layers]


def train(x: np.ndarray, c: np.ndarray, seed: int, hidden: int = 64, n_layers: int = 5, epochs: int = 60,
          batch: int = 4096, lr: float = 1e-3, val_frac: float = 0.05, log=print) -> tuple[MAF, dict]:
    """Fit a MAF to standardised (x, c); early-stops on a held-out fraction."""
    torch.manual_seed(seed)
    dev = device()
    n = x.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    nv = int(n * val_frac)
    xv, cv = (torch.tensor(a[perm[:nv]], dtype=torch.float32, device=dev) for a in (x, c))
    xt, ct = (torch.tensor(a[perm[nv:]], dtype=torch.float32, device=dev) for a in (x, c))
    model = MAF(x.shape[1], c.shape[1], hidden, n_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)  # fused update OOMs on shared-memory iGPUs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, history, bad = np.inf, None, [], 0
    for ep in range(epochs):
        model.train()
        order = torch.randperm(xt.shape[0], device=dev)
        for i in range(0, xt.shape[0], batch):
            idx = order[i:i + batch]
            loss = -model.log_prob(xt[idx], ct[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(-model.log_prob(xv, cv).mean())
        history.append(vl)
        if not np.isfinite(vl):
            raise FloatingPointError(f"non-finite validation loss at epoch {ep} (seed {seed})")
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  seed {seed} epoch {ep}: val NLL {vl:.4f}")
        if bad >= 8:
            break
    model.load_state_dict(best_state)
    return model, {"val_nll": best, "history": history, "device": str(dev), "epochs_run": len(history)}

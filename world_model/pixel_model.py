"""Pixel-space world model.

A small Dreamer-style autoencoder + latent dynamics:
  * Encoder: CNN maps a (3, H, W) frame to a latent vector ``z``.
  * Dynamics: MLP maps ``(z_t, a_t)`` to a *residual* update ``z_{t+1} = z_t + delta``.
    The delta head starts at zero so the model begins as "predict no change".
  * Decoder: transposed-CNN maps ``z`` back to a (3, H, W) frame.

Unlike the state-space model in ``model.py``, there's no analytic baseline
here — the network has to discover ball dynamics from pixels alone. The
training loop in ``examples/train_pixels.py`` jointly fits reconstruction,
latent dynamics (with a stop-grad on the encoder target to avoid collapse),
and decoded multi-step pixel prediction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """``(B, 3, 64, 64)`` -> ``(B, latent_dim)``."""

    def __init__(self, ch: int = 32, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.SiLU(),         # 64 -> 32
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),    # 32 -> 16
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.SiLU(),  # 16 -> 8
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1), nn.SiLU(),  # 8 -> 4
        )
        self.head = nn.Linear(ch * 4 * 4 * 4, latent_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h = self.net(img)
        return self.head(h.flatten(1))


class Decoder(nn.Module):
    """``(B, latent_dim)`` -> ``(B, 3, 64, 64)`` in [0, 1]."""

    def __init__(self, ch: int = 32, latent_dim: int = 64):
        super().__init__()
        self.ch4 = ch * 4
        self.head = nn.Linear(latent_dim, self.ch4 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.ch4, self.ch4, 4, 2, 1), nn.SiLU(),  # 4 -> 8
            nn.ConvTranspose2d(self.ch4, ch * 2, 4, 2, 1), nn.SiLU(),    # 8 -> 16
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU(),          # 16 -> 32
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),                          # 32 -> 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.head(z).view(-1, self.ch4, 4, 4)
        return torch.sigmoid(self.net(h))


class Dynamics(nn.Module):
    """Residual latent step: ``z_{t+1} = z_t + MLP(z_t, a_t)``.

    The output layer is zero-initialized so the model starts as identity
    and only learns deviations from "no change".
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        if self.action_dim > 0 and a is not None:
            x = torch.cat([z, a], dim=-1)
        else:
            x = z
        return z + self.net(x)


class PixelWorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 0,
        ch: int = 32,
        hidden: int = 128,
        state_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.encoder = Encoder(ch, latent_dim)
        self.decoder = Decoder(ch, latent_dim)
        self.dynamics = Dynamics(latent_dim, action_dim, hidden)
        # Optional: linear probe from latent to normalized state. Trained with an
        # auxiliary MSE loss so the latent is forced to carry position+velocity
        # info — without this, the encoder collapses to a constant under
        # joint dynamics training.
        self.state_head = nn.Linear(latent_dim, state_dim) if state_dim > 0 else None

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Accepts ``(..., 3, H, W)`` floats in [0,1]; returns ``(..., latent_dim)``."""
        *batch, c, h, w = frames.shape
        z = self.encoder(frames.reshape(-1, c, h, w))
        return z.reshape(*batch, -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Accepts ``(..., latent_dim)``; returns ``(..., 3, H, W)`` in [0,1]."""
        *batch, d = z.shape
        out = self.decoder(z.reshape(-1, d))
        c, h, w = out.shape[-3:]
        return out.reshape(*batch, c, h, w)

    def step(self, z: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        return self.dynamics(z, a)

    @torch.no_grad()
    def rollout(self, frame0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Autoregressive rollout in latent space, decoded back to frames.

        ``frame0`` is either ``(3, H, W)`` or ``(H, W, 3)``, uint8 or float.
        Returns ``(T+1, 3, H, W)`` floats in [0,1].
        """
        device = next(self.parameters()).device
        f = torch.as_tensor(frame0).to(device).float()
        if f.dtype != torch.float32 and f.max() > 1.5:
            pass  # already float, leave as-is
        if f.max() > 1.5:
            f = f / 255.0
        if f.shape[-1] == 3 and f.shape[0] != 3:
            f = f.permute(2, 0, 1)
        z = self.encoder(f.unsqueeze(0))
        out = [self.decoder(z).squeeze(0).cpu().numpy()]
        a_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
        for t in range(a_all.shape[0]):
            z = self.dynamics(z, a_all[t : t + 1])
            out.append(self.decoder(z).squeeze(0).cpu().numpy())
        return np.stack(out)


def load(path: str | Path, map_location: str | torch.device = "cpu") -> PixelWorldModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    model = PixelWorldModel(
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        ch=cfg["ch"],
        hidden=cfg["hidden"],
        state_dim=cfg.get("state_dim", 0),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

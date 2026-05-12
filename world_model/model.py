"""PyTorch state-space world model.

A small residual MLP that maps ``(state, action)`` to a predicted
``next_state``. Internally it predicts a *normalized state delta* and
adds it back to the input state, which is the standard parameterization
for stable, multi-step rollouts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DynamicsMLP(nn.Module):
    """Residual MLP world model.

    Parameters are kept in two groups:
      * ``net``: the learned weights.
      * normalization buffers (``state_mean``/``state_std`` and the same
        for the delta target). Stored on the module so a single
        ``state_dict()`` is enough to reproduce predictions later.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)

        # Filled in by ``set_norm`` before training/inference.
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("delta_mean", torch.zeros(state_dim))
        self.register_buffer("delta_std", torch.ones(state_dim))
        # Action gets a single scale, not per-dim — forces are isotropic.
        self.register_buffer("action_scale", torch.ones(1))

    def set_norm(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        delta_mean: np.ndarray,
        delta_std: np.ndarray,
        action_scale: float = 1.0,
    ) -> None:
        self.state_mean.copy_(torch.as_tensor(state_mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(state_std, dtype=torch.float32))
        self.delta_mean.copy_(torch.as_tensor(delta_mean, dtype=torch.float32))
        self.delta_std.copy_(torch.as_tensor(delta_std, dtype=torch.float32))
        self.action_scale.fill_(float(action_scale))

    def predict_delta_norm(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s = (state - self.state_mean) / self.state_std
        a = action / self.action_scale
        return self.net(torch.cat([s, a], dim=-1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next state (un-normalized)."""
        delta_norm = self.predict_delta_norm(state, action)
        delta = delta_norm * self.delta_std + self.delta_mean
        return state + delta

    @torch.no_grad()
    def rollout(
        self,
        state0: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Autoregressively roll forward from ``state0``.

        ``actions`` has shape ``(T, action_dim)``. Returns an array of
        shape ``(T+1, state_dim)`` starting with ``state0``.
        """
        device = self.state_mean.device
        s = torch.as_tensor(state0, dtype=torch.float32, device=device).unsqueeze(0)
        out = [s.squeeze(0).cpu().numpy()]
        a_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
        for t in range(a_all.shape[0]):
            s = self.forward(s, a_all[t : t + 1])
            out.append(s.squeeze(0).cpu().numpy())
        return np.stack(out)


def load(path: str | Path, map_location: str | torch.device = "cpu") -> DynamicsMLP:
    """Reconstruct a model from a checkpoint written by ``train.py``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    model = DynamicsMLP(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

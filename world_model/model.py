"""PyTorch state-space world model: residual over a free-flight baseline.

The network predicts a *correction* on top of an analytic ballistic
step (gravity + applied force, no collisions). This means the model
only has to learn the wall/ball-collision residual instead of
rediscovering Newtonian motion from data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def ballistic_step(
    state: torch.Tensor,
    action: torch.Tensor,
    dt: torch.Tensor | float,
    gravity_y: torch.Tensor | float,
    mass: torch.Tensor | float,
) -> torch.Tensor:
    """Free-flight one-step prediction for any number of balls.

    ``state`` has shape ``(..., 4N)`` laid out as ``[x, y, vx, vy]`` per
    ball; ``action`` has shape ``(..., 2N)`` as ``[fx, fy]`` per ball.
    """
    *batch, dim = state.shape
    n = dim // 4
    s = state.reshape(*batch, n, 4)
    a = action.reshape(*batch, n, 2)
    ax = a[..., 0] / mass
    ay = a[..., 1] / mass + gravity_y
    x = s[..., 0] + s[..., 2] * dt + 0.5 * ax * dt * dt
    y = s[..., 1] + s[..., 3] * dt + 0.5 * ay * dt * dt
    vx = s[..., 2] + ax * dt
    vy = s[..., 3] + ay * dt
    out = torch.stack([x, y, vx, vy], dim=-1)
    return out.reshape(*batch, dim)


class DynamicsMLP(nn.Module):
    """Residual-over-baseline dynamics model.

    ``forward(s, a) = ballistic_step(s, a) + correction(s, a)``

    The correction net is zero-initialized at the final layer so the
    model starts as pure ballistic motion and learns only the deviation
    (collisions) from data.
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
        final = nn.Linear(in_dim, state_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

        # Normalization buffers (filled by ``set_norm``).
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("c_scale", torch.ones(state_dim))
        self.register_buffer("action_scale", torch.ones(1))
        # Physics constants (filled by ``set_physics``).
        self.register_buffer("dt", torch.tensor(1.0 / 60.0))
        self.register_buffer("gravity_y", torch.tensor(900.0))
        self.register_buffer("mass", torch.tensor(1.0))

    def set_norm(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        c_scale: np.ndarray,
        action_scale: float = 1.0,
    ) -> None:
        self.state_mean.copy_(torch.as_tensor(state_mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(state_std, dtype=torch.float32))
        self.c_scale.copy_(torch.as_tensor(c_scale, dtype=torch.float32))
        self.action_scale.fill_(float(action_scale))

    def set_physics(self, dt: float, gravity_y: float, mass: float) -> None:
        self.dt.fill_(float(dt))
        self.gravity_y.fill_(float(gravity_y))
        self.mass.fill_(float(mass))

    def correction(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s_n = (state - self.state_mean) / self.state_std
        a_n = action / self.action_scale
        return self.net(torch.cat([s_n, a_n], dim=-1)) * self.c_scale

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        base = ballistic_step(state, action, self.dt, self.gravity_y, self.mass)
        return base + self.correction(state, action)

    @torch.no_grad()
    def rollout(self, state0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        device = self.state_mean.device
        s = torch.as_tensor(state0, dtype=torch.float32, device=device).unsqueeze(0)
        out = [s.squeeze(0).cpu().numpy()]
        a_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
        for t in range(a_all.shape[0]):
            s = self.forward(s, a_all[t : t + 1])
            out.append(s.squeeze(0).cpu().numpy())
        return np.stack(out)


def load(path: str | Path, map_location: str | torch.device = "cpu") -> DynamicsMLP:
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

"""PyTorch state-space world model: residual over a free-flight baseline.

The network has a shared trunk and two heads:
  * ``mean_head`` predicts a *correction* on top of an analytic ballistic
    step (gravity + applied force, no collisions).
  * ``logvar_head`` predicts a log-variance for each output dim,
    interpreted in normalized (per-``state_std``) units. Used only when
    the model is trained with Gaussian NLL.

``forward()`` returns the deterministic mean prediction.
``predict_with_var()`` returns ``(mean, log_var_normalized)`` for use
with the NLL loss and uncertainty reporting.
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
    """Free-flight one-step prediction. ``state`` ``(..., 4N)``, ``action`` ``(..., 2N)``."""
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


def reflect_walls(
    state: torch.Tensor,
    wall_min: torch.Tensor,
    wall_max: torch.Tensor,
    restitution: torch.Tensor | float,
) -> torch.Tensor:
    """Axis-aligned ball-wall reflection on a (..., 4N) state.

    Clamps each ball's position into ``[wall_min, wall_max]`` and flips the
    inward velocity component, scaled by ``restitution``. With ``wall_min`` /
    ``wall_max`` set to -/+inf this is a no-op, so old checkpoints behave
    identically until ``set_walls`` is called.
    """
    *batch, dim = state.shape
    n = dim // 4
    s = state.reshape(*batch, n, 4)
    pos = s[..., :2]
    vel = s[..., 2:]
    under = pos < wall_min
    over = pos > wall_max
    new_pos = torch.where(under, wall_min, pos)
    new_pos = torch.where(over, wall_max, new_pos)
    flip = (under & (vel < 0)) | (over & (vel > 0))
    new_vel = torch.where(flip, -restitution * vel, vel)
    out = torch.cat([new_pos, new_vel], dim=-1)
    return out.reshape(*batch, dim)


class DynamicsMLP(nn.Module):
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

        trunk: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for _ in range(n_layers):
            trunk += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        self.trunk = nn.Sequential(*trunk)

        # Two heads off the shared trunk.
        self.mean_head = nn.Linear(hidden, state_dim)
        self.logvar_head = nn.Linear(hidden, state_dim)
        # Start the correction at zero (pure baseline) and start the
        # predicted log-variance at zero (≈ unit variance in normalized
        # units, i.e. variance ~ state_std^2 in raw units).
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("c_scale", torch.ones(state_dim))
        self.register_buffer("action_scale", torch.ones(1))
        self.register_buffer("dt", torch.tensor(1.0 / 60.0))
        self.register_buffer("gravity_y", torch.tensor(900.0))
        self.register_buffer("mass", torch.tensor(1.0))
        # Wall bounds default to ±inf -> reflect_walls is a no-op until set.
        self.register_buffer("wall_min", torch.full((2,), float("-inf")))
        self.register_buffer("wall_max", torch.full((2,), float("inf")))
        self.register_buffer("wall_e", torch.tensor(0.0))

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

    def set_walls(self, wall_min, wall_max, restitution: float) -> None:
        self.wall_min.copy_(torch.as_tensor(wall_min, dtype=torch.float32))
        self.wall_max.copy_(torch.as_tensor(wall_max, dtype=torch.float32))
        self.wall_e.fill_(float(restitution))

    def _baseline(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        base = ballistic_step(state, action, self.dt, self.gravity_y, self.mass)
        return reflect_walls(base, self.wall_min, self.wall_max, self.wall_e)

    def _trunk(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s_n = (state - self.state_mean) / self.state_std
        a_n = action / self.action_scale
        return self.trunk(torch.cat([s_n, a_n], dim=-1))

    def correction(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._trunk(state, action)
        return self.mean_head(h) * self.c_scale

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._baseline(state, action) + self.correction(state, action)

    def predict_with_var(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(next_state_mean, log_var_normalized)``.

        ``log_var_normalized`` is in units where ``state_std`` is the
        natural scale: variance in raw state units is
        ``exp(log_var_normalized) * state_std**2``.
        """
        h = self._trunk(state, action)
        corr = self.mean_head(h) * self.c_scale
        log_var = self.logvar_head(h)
        return self._baseline(state, action) + corr, log_var

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


class GraphDynamics(nn.Module):
    """Permutation-equivariant residual world model.

    Each ball is a node; one round of pairwise message passing models
    interactions (intended to capture ball-ball collisions, which the
    flat-state ``DynamicsMLP`` had to learn coordinate-by-coordinate).
    Same external API as ``DynamicsMLP``: ``forward(state, action)``
    returns a flat ``(..., 4N)`` next-state prediction.
    """

    def __init__(self, n_balls: int, hidden: int = 64, msg_hidden: int = 64):
        super().__init__()
        self.n_balls = n_balls
        self.state_dim = 4 * n_balls
        self.action_dim = 2 * n_balls
        self.hidden = hidden
        self.msg_hidden = msg_hidden

        # Node features: per-ball [x, y, vx, vy, fx, fy] (normalized) -> hidden.
        self.node_enc = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # Edge features: [receiver_feats, sender_feats, rel_pos(2), rel_vel(2), dist(1)] -> msg.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 5, msg_hidden), nn.SiLU(),
            nn.Linear(msg_hidden, msg_hidden), nn.SiLU(),
        )
        dec_in = hidden + msg_hidden
        self.mean_head = nn.Linear(dec_in, 4)
        self.logvar_head = nn.Linear(dec_in, 4)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

        # Per-ball stats (shape (4,)) — same normalization applies to every ball.
        self.register_buffer("ball_state_mean", torch.zeros(4))
        self.register_buffer("ball_state_std", torch.ones(4))
        self.register_buffer("c_scale", torch.ones(4))
        self.register_buffer("action_scale", torch.ones(1))
        self.register_buffer("dt", torch.tensor(1.0 / 60.0))
        self.register_buffer("gravity_y", torch.tensor(900.0))
        self.register_buffer("mass", torch.tensor(1.0))
        self.register_buffer("wall_min", torch.full((2,), float("-inf")))
        self.register_buffer("wall_max", torch.full((2,), float("inf")))
        self.register_buffer("wall_e", torch.tensor(0.0))

    def set_norm(
        self,
        ball_state_mean: np.ndarray,
        ball_state_std: np.ndarray,
        c_scale: np.ndarray,
        action_scale: float = 1.0,
    ) -> None:
        self.ball_state_mean.copy_(torch.as_tensor(ball_state_mean, dtype=torch.float32))
        self.ball_state_std.copy_(torch.as_tensor(ball_state_std, dtype=torch.float32))
        self.c_scale.copy_(torch.as_tensor(c_scale, dtype=torch.float32))
        self.action_scale.fill_(float(action_scale))

    def set_physics(self, dt: float, gravity_y: float, mass: float) -> None:
        self.dt.fill_(float(dt))
        self.gravity_y.fill_(float(gravity_y))
        self.mass.fill_(float(mass))

    def set_walls(self, wall_min, wall_max, restitution: float) -> None:
        self.wall_min.copy_(torch.as_tensor(wall_min, dtype=torch.float32))
        self.wall_max.copy_(torch.as_tensor(wall_max, dtype=torch.float32))
        self.wall_e.fill_(float(restitution))

    def _baseline(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        base = ballistic_step(state, action, self.dt, self.gravity_y, self.mass)
        return reflect_walls(base, self.wall_min, self.wall_max, self.wall_e)

    def _trunk(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns per-ball decoder input ``(..., N, hidden + msg_hidden)``."""
        *batch_dims, dim = state.shape
        n = dim // 4
        s_per_ball = state.reshape(*batch_dims, n, 4)
        a_per_ball = action.reshape(*batch_dims, n, 2)

        s_n = (s_per_ball - self.ball_state_mean) / self.ball_state_std
        a_n = a_per_ball / self.action_scale
        node_feats = self.node_enc(torch.cat([s_n, a_n], dim=-1))  # (..., N, H)

        pos = s_per_ball[..., :2]
        vel = s_per_ball[..., 2:]
        # Pairwise indexed (..., receiver i, sender j, :). sender-minus-receiver.
        rel_pos = pos.unsqueeze(-3) - pos.unsqueeze(-2)
        rel_vel = vel.unsqueeze(-3) - vel.unsqueeze(-2)
        rel_pos_n = rel_pos / self.ball_state_std[:2]
        rel_vel_n = rel_vel / self.ball_state_std[2:]
        dist = rel_pos.norm(dim=-1, keepdim=True)
        dist_n = dist / (self.ball_state_std[:2].norm() + 1e-6)
        edge_feats = torch.cat([rel_pos_n, rel_vel_n, dist_n], dim=-1)  # (..., N, N, 5)

        target_shape = list(edge_feats.shape[:-1]) + [self.hidden]
        node_i = node_feats.unsqueeze(-2).expand(target_shape)  # receiver
        node_j = node_feats.unsqueeze(-3).expand(target_shape)  # sender
        msgs = self.edge_mlp(torch.cat([node_i, node_j, edge_feats], dim=-1))

        eye = torch.eye(n, dtype=torch.bool, device=msgs.device)
        msgs = msgs.masked_fill(eye.unsqueeze(-1), 0.0)
        agg = msgs.sum(dim=-2)  # aggregate over senders (..., N, msg_hidden)
        return torch.cat([node_feats, agg], dim=-1)

    def correction(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._trunk(state, action)
        corr = self.mean_head(h) * self.c_scale  # (..., N, 4)
        return corr.reshape(state.shape)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._baseline(state, action) + self.correction(state, action)

    def predict_with_var(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._trunk(state, action)
        corr = self.mean_head(h) * self.c_scale
        log_var = self.logvar_head(h)
        return (
            self._baseline(state, action) + corr.reshape(state.shape),
            log_var.reshape(state.shape),
        )

    @torch.no_grad()
    def rollout(self, state0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        device = self.ball_state_mean.device
        s = torch.as_tensor(state0, dtype=torch.float32, device=device).unsqueeze(0)
        out = [s.squeeze(0).cpu().numpy()]
        a_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
        for t in range(a_all.shape[0]):
            s = self.forward(s, a_all[t : t + 1])
            out.append(s.squeeze(0).cpu().numpy())
        return np.stack(out)


def load(path: str | Path, map_location: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    model_type = cfg.get("model_type", "mlp")
    if model_type == "graph":
        model = GraphDynamics(
            n_balls=cfg["n_balls"],
            hidden=cfg["hidden"],
            msg_hidden=cfg.get("msg_hidden", cfg["hidden"]),
        )
    else:
        model = DynamicsMLP(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            hidden=cfg["hidden"],
            n_layers=cfg["n_layers"],
        )
    # strict=False so checkpoints saved before wall buffers were added still load.
    model.load_state_dict(ckpt["state_dict"], strict=False)
    if "wall_min" in cfg and "wall_max" in cfg:
        model.set_walls(cfg["wall_min"], cfg["wall_max"], cfg.get("wall_e", 0.0))
    model.eval()
    return model

"""Minimal world model with two key improvements over a vanilla MLP:

  1. Residual over a free-flight baseline. We hard-code an analytic
     gravity-only step and have the network predict only a *correction*
     to it. The model no longer has to rediscover ballistic motion from
     scratch — its entire job is to learn the bounce term, which is
     mostly zero except near walls.

  2. k-step rollout loss. Instead of training on one-step MSE, the
     model is unrolled for k steps with predictions fed back as inputs
     and the loss is summed across all k steps. This directly penalizes
     compounding error at the horizon we actually evaluate at.

Run from the project root:
    python -m examples.mini
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from world_model.physics import Body, Wall, World, vec2

W, H = 160, 120
DT = 1.0 / 60.0
G = 600.0


def make_world() -> World:
    world = World(gravity=vec2(0.0, G))
    m = 8
    world.add_wall(Wall(vec2(m, m), vec2(W - m, m)))
    world.add_wall(Wall(vec2(W - m, m), vec2(W - m, H - m)))
    world.add_wall(Wall(vec2(W - m, H - m), vec2(m, H - m)))
    world.add_wall(Wall(vec2(m, H - m), vec2(m, m)))
    return world


def reset_ball(world: World, rng: np.random.Generator) -> None:
    world.bodies.clear()
    world.add_body(Body(
        pos=vec2(float(rng.uniform(20, W - 20)), float(rng.uniform(20, H - 40))),
        vel=vec2(float(rng.uniform(-80, 80)), float(rng.uniform(-40, 40))),
        radius=6.0,
    ))


def collect_episodes(n_episodes: int, steps: int, seed: int = 0) -> np.ndarray:
    """Trajectories of shape (n_episodes, steps + 1, 4)."""
    rng = np.random.default_rng(seed)
    world = make_world()
    out = np.empty((n_episodes, steps + 1, 4), dtype=np.float32)
    for ep in range(n_episodes):
        reset_ball(world, rng)
        out[ep, 0] = world.state_vector()
        for t in range(steps):
            world.step(DT)
            out[ep, t + 1] = world.state_vector()
    return out


def ballistic_step(s: torch.Tensor, dt: float = DT, g: float = G) -> torch.Tensor:
    """Free-flight (gravity-only) one-step prediction. Vectorized over s."""
    x = s[..., 0] + s[..., 2] * dt
    y = s[..., 1] + s[..., 3] * dt + 0.5 * g * dt * dt
    vx = s[..., 2]
    vy = s[..., 3] + g * dt
    return torch.stack([x, y, vx, vy], dim=-1)


class Mini(nn.Module):
    def __init__(self, dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        # Start the correction at ~0 so the model begins as pure free-flight.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("s_mean", torch.zeros(dim))
        self.register_buffer("s_std", torch.ones(dim))
        self.register_buffer("c_scale", torch.ones(dim))

    def correction(self, s: torch.Tensor) -> torch.Tensor:
        return self.net((s - self.s_mean) / self.s_std) * self.c_scale

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return ballistic_step(s) + self.correction(s)


def evaluate(model: Mini, n_eval: int = 20, horizon: int = 40, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    err = np.zeros(horizon + 1)
    for _ in range(n_eval):
        world = make_world()
        reset_ball(world, rng)
        s0 = world.state_vector().astype(np.float32)
        truth = [s0]
        for _ in range(horizon):
            world.step(DT)
            truth.append(world.state_vector().astype(np.float32))
        truth = np.stack(truth)
        with torch.no_grad():
            s = torch.from_numpy(s0).unsqueeze(0)
            pred = [s.squeeze(0).numpy()]
            for _ in range(horizon):
                s = model(s)
                pred.append(s.squeeze(0).numpy())
        pred = np.stack(pred)
        err += ((pred[:, :2] - truth[:, :2]) ** 2).sum(-1)
    return err / n_eval


def main() -> None:
    print("generating episodes...")
    eps = collect_episodes(n_episodes=200, steps=120)
    eps_t = torch.from_numpy(eps)
    n_eps, T_full, _ = eps_t.shape
    T = T_full - 1

    s_all = eps.reshape(-1, 4)
    s_in = eps[:, :-1].reshape(-1, 4)
    s_out = eps[:, 1:].reshape(-1, 4)
    base_np = ballistic_step(torch.from_numpy(s_in)).numpy()
    c_scale = (s_out - base_np).std(0) + 1e-6

    model = Mini()
    model.s_mean.copy_(torch.from_numpy(s_all.mean(0)))
    model.s_std.copy_(torch.from_numpy(s_all.std(0) + 1e-6))
    model.c_scale.copy_(torch.from_numpy(c_scale))

    rollout_len = 10
    batch = 128
    batches_per_epoch = 32
    epochs = 40

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    starts_max = T - rollout_len
    k_range = torch.arange(rollout_len + 1)

    print(f"training with {rollout_len}-step rollout loss...")
    for epoch in range(epochs):
        total = 0.0
        for _ in range(batches_per_epoch):
            ei = torch.randint(0, n_eps, (batch,))
            si = torch.randint(0, starts_max, (batch,))
            time_idx = si[:, None] + k_range[None, :]
            batch_idx = ei[:, None].expand(-1, rollout_len + 1)
            seq = eps_t[batch_idx, time_idx]  # (B, k+1, 4)

            s_pred = seq[:, 0]
            loss = 0.0
            for k in range(rollout_len):
                s_pred = model(s_pred)
                loss = loss + ((s_pred - seq[:, k + 1]) ** 2).mean()
            loss = loss / rollout_len

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:02d}  rollout_mse={total / batches_per_epoch:.4f}")

    # For comparison: how well does the baseline-only predictor do?
    class Baseline(nn.Module):
        def forward(self, s):
            return ballistic_step(s)
    baseline_err = evaluate(Baseline())
    model_err = evaluate(model)

    print("\nper-step position MSE (vs simulator), averaged over 20 episodes:")
    print("  horizon | baseline (ballistic only) | learned model")
    for t in (1, 5, 10, 20, 40):
        print(f"     t={t:2d} | {baseline_err[t]:>20.2f}     | {model_err[t]:>10.2f}")


if __name__ == "__main__":
    main()

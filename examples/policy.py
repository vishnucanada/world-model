"""Train a reach-a-target policy by backpropagating through the world model.

Demonstrates the "model as differentiable simulator" idea: a small MLP
policy maps ``(state, goal)`` to a 2D force, is trained by unrolling
through the learned 1-ball world model, and is then evaluated in the
*real* simulator to check that the learned dynamics transfer.

Pipeline:
    python -m examples.train --regenerate --n-balls 1 --force-scale 300 \\
        --stochastic --hidden 192 --data data/transitions_1b.npz \\
        --out checkpoints/world_model_1b.pt
    python -m examples.policy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from world_model.env import PhysicsEnv
from world_model.model import load as load_model


class Policy(nn.Module):
    def __init__(self, state_dim: int = 4, goal_dim: int = 2, hidden: int = 64, max_force: float = 1000.0):
        super().__init__()
        self.max_force = max_force
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.max_force * torch.tanh(self.net(x))


def sample_initial_state(n: int, w: float, h: float, margin: float = 25.0) -> torch.Tensor:
    pos_x = torch.rand(n) * (w - 2 * margin) + margin
    pos_y = torch.rand(n) * (h - 2 * margin) + margin
    vel = torch.randn(n, 2) * 20.0
    return torch.cat([torch.stack([pos_x, pos_y], dim=-1), vel], dim=-1)


def sample_goal(n: int, w: float, h: float, margin: float = 35.0) -> torch.Tensor:
    return torch.stack(
        [
            torch.rand(n) * (w - 2 * margin) + margin,
            torch.rand(n) * (h - 2 * margin) + margin,
        ],
        dim=-1,
    )


def eval_in_sim(
    policy: nn.Module,
    env: PhysicsEnv,
    n_trials: int,
    horizon: int,
    width: float,
    height: float,
    success_radius: float,
    seed: int,
    radius: float = 12.0,
) -> dict:
    rng = np.random.default_rng(seed)
    margin = radius + 12.0
    finals = []
    closest = []
    successes = 0
    for _ in range(n_trials):
        env.world.bodies.clear()
        env.add_ball(
            x=float(rng.uniform(margin, width - margin)),
            y=float(rng.uniform(margin, height - margin)),
            vx=float(rng.normal(0.0, 20.0)),
            vy=float(rng.normal(0.0, 20.0)),
            radius=radius,
        )
        gx = float(rng.uniform(35.0, width - 35.0))
        gy = float(rng.uniform(35.0, height - 35.0))
        goal_t = torch.tensor([[gx, gy]], dtype=torch.float32)
        best_d = float("inf")
        for _ in range(horizon):
            s_np = env.world.state_vector().astype(np.float32)
            with torch.no_grad():
                a_t = policy(torch.from_numpy(s_np).unsqueeze(0), goal_t)
            env.step(action=a_t.squeeze(0).numpy().reshape(1, 2))
            pos = env.world.state_vector()[:2]
            best_d = min(best_d, float(np.hypot(pos[0] - gx, pos[1] - gy)))
        s_final = env.world.state_vector()
        final = float(np.hypot(s_final[0] - gx, s_final[1] - gy))
        finals.append(final)
        closest.append(best_d)
        if final < success_radius:
            successes += 1
    return {
        "success_rate": successes / n_trials,
        "mean_final": float(np.mean(finals)),
        "median_final": float(np.median(finals)),
        "mean_closest": float(np.mean(closest)),
        "median_closest": float(np.median(closest)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/world_model_1b.pt"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/policy.pt"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batches-per-epoch", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rollout-T", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--max-force", type=float, default=2000.0)
    p.add_argument("--ctrl-weight", type=float, default=1e-5)
    p.add_argument("--near-goal-vel-weight", type=float, default=0.0,
                   help="weight on velocity penalty, multiplied by proximity-to-goal Gaussian")
    p.add_argument("--near-goal-scale", type=float, default=900.0,
                   help="length-scale^2 of the near-goal proximity Gaussian (units px^2)")
    p.add_argument("--eval-trials", type=int, default=20)
    p.add_argument("--eval-horizon", type=int, default=80)
    p.add_argument("--success-radius", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5, help="run a quick sim eval every N epochs and keep the best policy")
    p.add_argument("--eval-during-trials", type=int, default=10, help="trials per quick eval")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world = load_model(args.ckpt)
    for p_ in world.parameters():
        p_.requires_grad_(False)
    world.eval()

    cfg = torch.load(args.ckpt, weights_only=False)["config"]
    W = float(cfg["width"])
    H = float(cfg["height"])
    if cfg["state_dim"] != 4:
        raise ValueError(f"policy demo expects a 1-ball model (state_dim=4), got {cfg['state_dim']}")

    policy = Policy(max_force=args.max_force)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    eval_env = PhysicsEnv(width=int(W), height=int(H), headless=True)
    import copy
    best_score = float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    best_summary = None

    print(f"training policy with {args.rollout_T}-step rollouts through world model...")
    for epoch in range(args.epochs):
        total_goal = 0.0
        total_ctrl = 0.0
        for _ in range(args.batches_per_epoch):
            s = sample_initial_state(args.batch_size, W, H)
            g = sample_goal(args.batch_size, W, H)

            goal_loss = 0.0
            ctrl_loss = 0.0
            near_vel_loss = 0.0
            for t in range(args.rollout_T):
                a = policy(s, g)
                s = world(s, a)
                dist2 = ((s[..., :2] - g) ** 2).sum(-1)
                vel2 = (s[..., 2:] ** 2).sum(-1)
                near = torch.exp(-dist2 / args.near_goal_scale)
                goal_loss = goal_loss + dist2.mean()
                near_vel_loss = near_vel_loss + (near * vel2).mean()
                ctrl_loss = ctrl_loss + (a ** 2).mean()
            loss = (
                goal_loss / args.rollout_T
                + args.near_goal_vel_weight * near_vel_loss / args.rollout_T
                + args.ctrl_weight * ctrl_loss / args.rollout_T
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total_goal += (goal_loss / args.rollout_T).item()
            total_ctrl += (ctrl_loss / args.rollout_T).item()

        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            quick = eval_in_sim(
                policy, eval_env,
                n_trials=args.eval_during_trials,
                horizon=args.eval_horizon,
                width=W, height=H,
                success_radius=args.success_radius,
                seed=args.seed + 9999,
            )
            score = quick["mean_final"]
            tag = ""
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(policy.state_dict())
                best_summary = quick
                tag = "  <- best"
            print(
                f"  epoch {epoch + 1:02d}/{args.epochs}  "
                f"goal_mse={total_goal / args.batches_per_epoch:7.1f}  "
                f"sim_final_mean={score:6.1f}px  "
                f"success={100.0 * quick['success_rate']:5.1f}%{tag}"
            )

    # Restore best policy weights.
    policy.load_state_dict(best_state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "max_force": args.max_force}, args.out)
    print(f"saved best policy -> {args.out}")

    print(f"\nfinal eval in REAL simulator on {args.eval_trials} trials...")
    final_eval = eval_in_sim(
        policy, eval_env,
        n_trials=args.eval_trials,
        horizon=args.eval_horizon,
        width=W, height=H,
        success_radius=args.success_radius,
        seed=args.seed + 100,
    )
    eval_env.close()
    sr = final_eval["success_rate"]
    print(f"  success rate (final dist < {args.success_radius:.0f}px): "
          f"{int(round(sr * args.eval_trials))}/{args.eval_trials} ({100.0 * sr:.0f}%)")
    print(f"  final distance:    mean {final_eval['mean_final']:6.1f} px   median {final_eval['median_final']:6.1f} px")
    print(f"  closest approach:  mean {final_eval['mean_closest']:6.1f} px   median {final_eval['median_closest']:6.1f} px")


if __name__ == "__main__":
    main()

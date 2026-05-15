"""Train the state-space world model with multi-step rollout loss.

Key training choices:
  * residual-over-baseline model (see ``world_model.model``)
  * curriculum: rollout-len ramps from ``--rollout-len-start`` to
    ``--rollout-len`` linearly over the first half of training so the
    network learns one-step dynamics cleanly before being held
    accountable for long chains.
  * per-dim normalized loss: MSE is taken in units of ``state_std`` so
    position and velocity dimensions contribute proportionally and
    long-horizon errors don't drown out short-horizon ones.
  * cosine learning-rate annealing.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from world_model.dataset import SceneSpec, generate
from world_model.model import DynamicsMLP, GraphDynamics, ballistic_step, reflect_walls


def load_transitions(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/world_model.pt"))
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--n-balls", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches-per-epoch", type=int, default=64)
    p.add_argument("--rollout-len", type=int, default=16, help="max rollout length")
    p.add_argument("--rollout-len-start", type=int, default=3, help="initial rollout length for curriculum")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--model", choices=["mlp", "graph"], default="mlp",
                   help="mlp = flat-state DynamicsMLP; graph = permutation-equivariant interaction net")
    p.add_argument("--msg-hidden", type=int, default=64, help="message-MLP hidden width (graph only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--regenerate", action="store_true", help="Force dataset regeneration")
    p.add_argument("--force-scale", type=float, default=0.0, help="std of random forces injected during data generation")
    p.add_argument("--stochastic", action="store_true", help="train with Gaussian NLL using the variance head")
    args = p.parse_args()

    if args.regenerate or not args.data.exists():
        print(f"generating dataset -> {args.data}")
        info = generate(
            args.data,
            n_episodes=args.episodes,
            steps_per_episode=args.steps,
            spec=SceneSpec(n_balls=args.n_balls),
            force_scale=args.force_scale,
            seed=args.seed,
        )
        print(info)

    data = load_transitions(args.data)
    states = data["states"]
    actions = data["actions"]
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    dt = float(data["dt"])
    gravity_y = float(data["gravity_y"])
    mass = float(data["mass"])

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    n_balls = state_dim // 4
    if args.model == "graph":
        model = GraphDynamics(
            n_balls=n_balls, hidden=args.hidden, msg_hidden=args.msg_hidden
        ).to(device)
    else:
        model = DynamicsMLP(
            state_dim, action_dim, hidden=args.hidden, n_layers=args.layers
        ).to(device)
    model.set_physics(dt=dt, gravity_y=gravity_y, mass=mass)

    radius = float(data["radius"])
    restitution = float(data["restitution"])
    wall_margin = float(data["wall_margin"])
    width = float(data["width"])
    height = float(data["height"])
    inner = wall_margin + radius
    wall_min = [inner, inner]
    wall_max = [width - inner, height - inner]
    model.set_walls(wall_min, wall_max, restitution)

    s_flat = torch.from_numpy(states[:, :-1].reshape(-1, state_dim))
    sn_flat = torch.from_numpy(states[:, 1:].reshape(-1, state_dim))
    a_flat = torch.from_numpy(actions.reshape(-1, action_dim))
    with torch.no_grad():
        base = ballistic_step(
            s_flat,
            a_flat,
            torch.tensor(dt),
            torch.tensor(gravity_y),
            torch.tensor(mass),
        )
        base = reflect_walls(
            base,
            torch.tensor(wall_min, dtype=torch.float32),
            torch.tensor(wall_max, dtype=torch.float32),
            restitution,
        )
        residual = (sn_flat - base).numpy()
    action_scale = float(np.abs(actions).mean() + 1e-3) if actions.any() else 1.0

    if args.model == "graph":
        # Per-ball stats — every ball is identical so we share one (4,) vector.
        flat_per_ball = states.reshape(-1, 4)
        ball_mean = flat_per_ball.mean(axis=0)
        ball_std = flat_per_ball.std(axis=0) + 1e-6
        c_scale_per_ball = residual.reshape(-1, 4).std(axis=0) + 1e-6
        model.set_norm(ball_mean, ball_std, c_scale_per_ball, action_scale)
    else:
        c_scale = residual.std(axis=0) + 1e-6
        model.set_norm(
            state_mean=data["state_mean"],
            state_std=data["state_std"],
            c_scale=c_scale,
            action_scale=action_scale,
        )

    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    n_eps = states_t.shape[0]
    T = actions_t.shape[1]
    K_max = args.rollout_len
    K_start = max(1, min(args.rollout_len_start, K_max))
    if K_max > T:
        raise ValueError(f"rollout-len {K_max} exceeds episode length {T}")

    # Loss normalization: errors measured in per-dim "fractions of std".
    loss_scale = torch.as_tensor(data["state_std"], dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * args.batches_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr_min)

    ramp_epochs = max(1, args.epochs // 2)
    print(
        f"training {args.epochs} epochs, curriculum K={K_start}->{K_max} "
        f"over {ramp_epochs} epochs, normalized loss + cosine LR"
    )
    for epoch in range(args.epochs):
        # Linear ramp during the first half, then hold at K_max.
        t = min(1.0, epoch / max(1, ramp_epochs - 1))
        K = int(round(K_start + t * (K_max - K_start)))
        starts_max = T - K + 1
        k_state = torch.arange(K + 1, device=device)
        k_act = torch.arange(K, device=device)

        total = 0.0
        for _ in range(args.batches_per_epoch):
            ei = torch.randint(0, n_eps, (args.batch_size,), device=device)
            si = torch.randint(0, starts_max, (args.batch_size,), device=device)
            seq = states_t[ei[:, None].expand(-1, K + 1), si[:, None] + k_state[None, :]]
            act_seq = actions_t[ei[:, None].expand(-1, K), si[:, None] + k_act[None, :]]

            s_pred = seq[:, 0]
            loss = 0.0
            for k in range(K):
                if args.stochastic:
                    s_pred, log_var = model.predict_with_var(s_pred, act_seq[:, k])
                    log_var = torch.clamp(log_var, min=-8.0, max=8.0)
                    err_n = (s_pred - seq[:, k + 1]) / loss_scale
                    # Gaussian NLL in normalized units (constant term dropped).
                    nll = 0.5 * (err_n ** 2 * torch.exp(-log_var) + log_var)
                    loss = loss + nll.mean()
                else:
                    s_pred = model(s_pred, act_seq[:, k])
                    err_n = (s_pred - seq[:, k + 1]) / loss_scale
                    loss = loss + (err_n ** 2).mean()
            loss = loss / K

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"  epoch {epoch + 1:02d}/{args.epochs}  K={K:2d}  lr={lr_now:.2e}  "
                f"norm_mse={total / args.batches_per_epoch:.4f}"
            )

    if args.stochastic:
        # Calibration check: do steps with high predicted variance have high actual error?
        model.eval()
        with torch.no_grad():
            s_sample = s_flat[:5000].to(device)
            a_sample = a_flat[:5000].to(device)
            tgt_sample = sn_flat[:5000].to(device)
            mean_p, log_var = model.predict_with_var(s_sample, a_sample)
            log_var = torch.clamp(log_var, min=-8.0, max=8.0)
            err2_n = (((mean_p - tgt_sample) / loss_scale) ** 2).mean(dim=-1)
            std_n = torch.exp(0.5 * log_var.mean(dim=-1))
            # Rank steps by predicted std and bucket into quintiles.
            order = std_n.argsort()
            buckets = err2_n[order].view(5, -1).mean(dim=1).sqrt()
            std_buckets = std_n[order].view(5, -1).mean(dim=1)
            print("\nuncertainty calibration (5 quintiles by predicted std):")
            print("  bucket | mean predicted std | mean actual RMSE (normalized)")
            for i in range(5):
                print(f"     Q{i + 1}  | {std_buckets[i].item():>16.3f}   | {buckets[i].item():>10.3f}")
        model.train()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "model_type": args.model,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden": args.hidden,
                "n_layers": args.layers,
                "msg_hidden": args.msg_hidden,
                "n_balls": int(data["n_balls"]),
                "width": int(data["width"]),
                "height": int(data["height"]),
                "dt": dt,
                "gravity_y": gravity_y,
                "mass": mass,
                "wall_min": wall_min,
                "wall_max": wall_max,
                "wall_e": restitution,
            },
        },
        args.out,
    )
    print(f"saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()

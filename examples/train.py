"""Train the state-space world model with multi-step rollout loss.

Generates a transition dataset (if missing), fits a residual-over-baseline
MLP using k-step rollout MSE (the model is unrolled k steps, predictions
fed back as inputs, loss summed across all k steps), and saves a
self-contained checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model.dataset import SceneSpec, generate
from world_model.model import DynamicsMLP, ballistic_step


def load_transitions(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/world_model.pt"))
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--n-balls", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches-per-epoch", type=int, default=64)
    p.add_argument("--rollout-len", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--regenerate", action="store_true", help="Force dataset regeneration")
    args = p.parse_args()

    if args.regenerate or not args.data.exists():
        print(f"generating dataset -> {args.data}")
        info = generate(
            args.data,
            n_episodes=args.episodes,
            steps_per_episode=args.steps,
            spec=SceneSpec(n_balls=args.n_balls),
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

    model = DynamicsMLP(state_dim, action_dim, hidden=args.hidden, n_layers=args.layers).to(device)
    model.set_physics(dt=dt, gravity_y=gravity_y, mass=mass)

    # Correction-target scale: std of (truth - ballistic) over all one-step transitions.
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
        c_scale = (sn_flat - base).std(dim=0).numpy() + 1e-6
    action_scale = float(np.abs(actions).mean() + 1e-3) if actions.any() else 1.0
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
    K = args.rollout_len
    if K > T:
        raise ValueError(f"rollout-len {K} exceeds episode length {T}")
    starts_max = T - K + 1
    k_state = torch.arange(K + 1, device=device)
    k_act = torch.arange(K, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"training {args.epochs} epochs, {K}-step rollout loss...")
    for epoch in range(args.epochs):
        total = 0.0
        for _ in range(args.batches_per_epoch):
            ei = torch.randint(0, n_eps, (args.batch_size,), device=device)
            si = torch.randint(0, starts_max, (args.batch_size,), device=device)
            seq = states_t[ei[:, None].expand(-1, K + 1), si[:, None] + k_state[None, :]]
            act_seq = actions_t[ei[:, None].expand(-1, K), si[:, None] + k_act[None, :]]

            s_pred = seq[:, 0]
            loss = 0.0
            for k in range(K):
                s_pred = model(s_pred, act_seq[:, k])
                loss = loss + ((s_pred - seq[:, k + 1]) ** 2).mean()
            loss = loss / K

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:02d}/{args.epochs}  rollout_mse={total / args.batches_per_epoch:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden": args.hidden,
                "n_layers": args.layers,
                "n_balls": int(data["n_balls"]),
                "width": int(data["width"]),
                "height": int(data["height"]),
                "dt": dt,
                "gravity_y": gravity_y,
                "mass": mass,
            },
        },
        args.out,
    )
    print(f"saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()

"""Train the state-space world model.

Generates a transition dataset (if missing), fits a residual MLP to
predict the normalized state delta, and saves a checkpoint that bundles
the weights, normalization stats, and config.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from world_model.dataset import SceneSpec, generate
from world_model.model import DynamicsMLP


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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--regenerate", action="store_true", help="Force dataset regeneration")
    args = p.parse_args()

    if args.regenerate or not args.data.exists():
        print(f"generating dataset -> {args.data}")
        spec = SceneSpec(n_balls=args.n_balls)
        info = generate(
            args.data,
            n_episodes=args.episodes,
            steps_per_episode=args.steps,
            spec=spec,
            seed=args.seed,
        )
        print(info)

    data = load_transitions(args.data)
    states = torch.from_numpy(data["states"])
    actions = torch.from_numpy(data["actions"])
    next_states = torch.from_numpy(data["next_states"])
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    torch.manual_seed(args.seed)
    n = states.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TensorDataset(states[train_idx], actions[train_idx], next_states[train_idx])
    val_ds = TensorDataset(states[val_idx], actions[val_idx], next_states[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = DynamicsMLP(state_dim, action_dim, hidden=args.hidden, n_layers=args.layers).to(device)
    action_scale = float(np.abs(data["actions"]).mean() + 1e-3) if data["actions"].any() else 1.0
    model.set_norm(
        state_mean=data["state_mean"],
        state_std=data["state_std"],
        delta_mean=data["delta_mean"],
        delta_std=data["delta_std"],
        action_scale=action_scale,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def step_loss(s, a, sn):
        # Train on the normalized delta so the loss scale is well-behaved.
        target = (sn - s - model.delta_mean) / model.delta_std
        pred = model.predict_delta_norm(s, a)
        return torch.nn.functional.mse_loss(pred, target)

    def eval_one_step():
        model.eval()
        total = 0.0
        n_seen = 0
        with torch.no_grad():
            for s, a, sn in val_loader:
                s, a, sn = s.to(device), a.to(device), sn.to(device)
                pred_next = model(s, a)
                total += torch.nn.functional.mse_loss(pred_next, sn, reduction="sum").item()
                n_seen += sn.numel()
        model.train()
        return total / max(n_seen, 1)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for s, a, sn in train_loader:
            s, a, sn = s.to(device), a.to(device), sn.to(device)
            loss = step_loss(s, a, sn)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        val_mse = eval_one_step()
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}  "
            f"train_norm_mse={train_loss:.5f}  val_state_mse={val_mse:.4f}"
        )

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
                "dt": float(data["dt"]),
            },
        },
        args.out,
    )
    print(f"saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()

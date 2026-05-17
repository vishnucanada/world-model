"""Train the pixel-space world model.

Joint loss:
  L = L_recon + alpha * L_dyn + beta * L_pred
where
  L_recon: autoencoder MSE on every frame in the batch
  L_dyn  : MSE between predicted latents and encoder latents (stop-grad on
           the target so the encoder isn't pushed toward a degenerate constant)
  L_pred : MSE between decoded predicted frames and ground-truth frames,
           autoregressively unrolled for K steps in latent space.

Frames are stored as uint8 on CPU and converted to float per batch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from world_model.dataset import SceneSpec, generate
from world_model.pixel_model import PixelWorldModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/transitions_pixels.npz"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/world_model_pixels.pt"))
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--n-balls", type=int, default=5)
    p.add_argument("--frame-w", type=int, default=64)
    p.add_argument("--frame-h", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--ch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batches-per-epoch", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rollout-len", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=1.0, help="latent dynamics loss weight")
    p.add_argument("--beta", type=float, default=1.0, help="pixel prediction loss weight")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.regenerate or not args.data.exists():
        print(f"generating pixel dataset -> {args.data}")
        info = generate(
            args.data,
            n_episodes=args.episodes,
            steps_per_episode=args.steps,
            spec=SceneSpec(n_balls=args.n_balls),
            frame_size=(args.frame_w, args.frame_h),
            seed=args.seed,
        )
        print(info)

    with np.load(args.data) as f:
        frames_u8 = f["frames"]                  # (N, T+1, H, W, 3) uint8
        actions_np = f["actions"]                # (N, T, action_dim) float32
    n_eps, T_plus_1, fh, fw, _ = frames_u8.shape
    T = T_plus_1 - 1
    action_dim = int(actions_np.shape[-1])

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Keep frames as uint8 on CPU to save memory; convert to float per batch.
    frames_t = torch.from_numpy(frames_u8).permute(0, 1, 4, 2, 3).contiguous()  # (N, T+1, 3, H, W)
    actions_t = torch.from_numpy(actions_np)

    model = PixelWorldModel(
        latent_dim=args.latent_dim,
        action_dim=action_dim,
        ch=args.ch,
        hidden=args.hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params  (latent_dim={args.latent_dim}, ch={args.ch})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    K = args.rollout_len
    starts_max = T - K + 1
    if starts_max <= 0:
        raise ValueError(f"rollout-len {K} too large for episode length {T}")

    print(f"training {args.epochs} epochs, K={K}, batch={args.batch_size}")
    for epoch in range(args.epochs):
        tot_r = tot_d = tot_p = 0.0
        for _ in range(args.batches_per_epoch):
            ei = torch.randint(0, n_eps, (args.batch_size,))
            si = torch.randint(0, starts_max, (args.batch_size,))
            k_state = torch.arange(K + 1)
            k_act = torch.arange(K)
            frame_seq = frames_t[ei[:, None], si[:, None] + k_state[None, :]]  # (B, K+1, 3, H, W) uint8
            act_seq = actions_t[ei[:, None], si[:, None] + k_act[None, :]].to(device)

            f_in = frame_seq.to(device, dtype=torch.float32) / 255.0
            B = f_in.shape[0]

            z_seq = model.encode(f_in)                  # (B, K+1, latent)
            recon = model.decode(z_seq)
            L_recon = F.mse_loss(recon, f_in)

            z_pred = [z_seq[:, 0]]
            for k in range(K):
                z_pred.append(model.step(z_pred[-1], act_seq[:, k]))
            z_pred = torch.stack(z_pred, dim=1)         # (B, K+1, latent)

            L_dyn = F.mse_loss(z_pred[:, 1:], z_seq[:, 1:].detach())
            pred_frames = model.decode(z_pred[:, 1:])
            L_pred = F.mse_loss(pred_frames, f_in[:, 1:])

            L = L_recon + args.alpha * L_dyn + args.beta * L_pred
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_r += L_recon.item(); tot_d += L_dyn.item(); tot_p += L_pred.item()

        n_b = args.batches_per_epoch
        print(
            f"  epoch {epoch+1:02d}/{args.epochs}  "
            f"recon={tot_r/n_b:.4f}  dyn={tot_d/n_b:.4f}  pred={tot_p/n_b:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "latent_dim": args.latent_dim,
                "action_dim": action_dim,
                "ch": args.ch,
                "hidden": args.hidden,
                "frame_w": args.frame_w,
                "frame_h": args.frame_h,
                "n_balls": args.n_balls,
            },
        },
        args.out,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

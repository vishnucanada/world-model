"""Visualize the pixel world model: GT vs autoregressive model rollout.

Runs a fresh episode in the real simulator, takes its first frame, and lets
the model dream forward in latent space for ``--horizon`` steps. Saves a
side-by-side animated GIF (truth | model).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from world_model.dataset import SceneSpec, _reset_scene, _render_resized
from world_model.env import PhysicsEnv
from world_model.pixel_model import load as load_pixel_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/world_model_pixels.pt"))
    p.add_argument("--out", type=Path, default=Path("viz/pixels.gif"))
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    model = load_pixel_model(args.ckpt)
    cfg = torch.load(args.ckpt, weights_only=False)["config"]
    fw, fh = int(cfg["frame_w"]), int(cfg["frame_h"])
    n_balls = int(cfg["n_balls"])
    action_dim = int(cfg["action_dim"])

    env = PhysicsEnv(width=320, height=240, headless=True)
    rng = np.random.default_rng(args.seed)
    _reset_scene(env, SceneSpec(n_balls=n_balls, width=320, height=240), rng)

    gt_frames = []
    env.renderer.draw(env.world)
    gt_frames.append(_render_resized(env, (fw, fh)))
    for _ in range(args.horizon):
        env.step(action=None)
        gt_frames.append(_render_resized(env, (fw, fh)))
    env.close()

    actions = np.zeros((args.horizon, action_dim), dtype=np.float32)
    pred = model.rollout(gt_frames[0], actions)  # (T+1, 3, H, W) float[0,1]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    up = args.upscale
    images = []
    for t, (gt, pr) in enumerate(zip(gt_frames, pred)):
        pr_u8 = (pr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        gap = np.full((gt.shape[0], 4, 3), 60, dtype=np.uint8)
        panel = np.concatenate([gt, gap, pr_u8], axis=1)
        img = Image.fromarray(panel).resize(
            (panel.shape[1] * up, panel.shape[0] * up), Image.NEAREST
        )
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), "truth", fill=(255, 255, 255))
        draw.text((gt.shape[1] * up + 8, 4), "model", fill=(255, 255, 255))
        draw.text((img.size[0] - 50, 4), f"t={t}", fill=(200, 200, 200))
        images.append(img)

    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=False,
    )
    print(f"saved -> {args.out}  ({len(images)} frames @ {args.fps} fps)")

    # Numeric: per-step pixel MSE between truth and prediction.
    gt_f = np.stack(gt_frames).astype(np.float32) / 255.0  # (T+1, H, W, 3)
    pr_f = pred.transpose(0, 2, 3, 1)                       # (T+1, H, W, 3)
    mse_per_t = ((gt_f - pr_f) ** 2).mean(axis=(1, 2, 3))
    print("per-step pixel MSE (lower is better):")
    for t in [0, 1, 5, 10, 20, 40, args.horizon]:
        if t < len(mse_per_t):
            print(f"  t={t:3d}: {mse_per_t[t]:.4f}")


if __name__ == "__main__":
    main()

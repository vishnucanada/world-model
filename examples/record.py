"""Headless rollout. Demonstrates the multi-modal observation interface:
each step yields both a state vector and an RGB frame, ready for a world model.
"""
from pathlib import Path

import numpy as np

from world_model import PhysicsEnv

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def main() -> None:
    env = PhysicsEnv(width=320, height=240, headless=True)
    rng = np.random.default_rng(0)
    for _ in range(5):
        env.add_ball(
            x=float(rng.uniform(50, 270)),
            y=float(rng.uniform(40, 100)),
            vx=float(rng.uniform(-100, 100)),
            vy=0.0,
            radius=float(rng.uniform(8, 14)),
            color=tuple(int(c) for c in rng.integers(120, 255, size=3)),
        )

    out_dir = Path("rollout_frames")
    out_dir.mkdir(exist_ok=True)
    states = []
    frames = []

    for t in range(120):
        obs = env.step()
        states.append(obs["state"])
        if t % 4 == 0:
            frames.append(obs["frame"])
            if HAS_PIL:
                Image.fromarray(obs["frame"]).save(out_dir / f"frame_{t:04d}.png")

    states = np.stack(states)
    np.save(out_dir / "states.npy", states)
    print(f"steps: {states.shape[0]}, state dim per step: {states.shape[1]}")
    print(f"sampled frames: {len(frames)} of shape {frames[0].shape}")
    if HAS_PIL:
        print(f"saved PNGs and states.npy to {out_dir}/")
    else:
        print("install pillow to save PNGs")
    env.close()


if __name__ == "__main__":
    main()

"""Evaluate the learned world model against the simulator.

Rolls out the model autoregressively from a shared initial state and
compares to the simulator's ground-truth trajectory. Reports MSE on
positions and velocities as a function of horizon. Optionally saves a
side-by-side image of predicted vs true frames for one episode.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model.dataset import SceneSpec, _reset_scene
from world_model.env import PhysicsEnv
from world_model.model import load as load_model


def _split_state(state: np.ndarray, n_balls: int) -> tuple[np.ndarray, np.ndarray]:
    s = state.reshape(n_balls, 4)
    return s[:, :2], s[:, 2:]


def _set_world_to_state(env: PhysicsEnv, state: np.ndarray) -> None:
    n = len(env.world.bodies)
    s = state.reshape(n, 4)
    for body, row in zip(env.world.bodies, s):
        body.pos = row[:2].astype(np.float64).copy()
        body.vel = row[2:].astype(np.float64).copy()


def run_episode(
    env: PhysicsEnv,
    model,
    spec: SceneSpec,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (true_traj, pred_traj), each (horizon+1, state_dim)."""
    _reset_scene(env, spec, rng)
    s0 = env.world.state_vector().astype(np.float32)

    # Ground truth from the simulator.
    true_traj = [s0.copy()]
    for _ in range(horizon):
        env.step(action=None)
        true_traj.append(env.world.state_vector().astype(np.float32))
    true_traj = np.stack(true_traj)

    # Model rollout from the same s0, zero actions.
    actions = np.zeros((horizon, 2 * spec.n_balls), dtype=np.float32)
    pred_traj = model.rollout(s0, actions)
    return true_traj, pred_traj


def horizon_errors(
    true_trajs: list[np.ndarray],
    pred_trajs: list[np.ndarray],
    n_balls: int,
) -> dict[str, np.ndarray]:
    true = np.stack(true_trajs)  # (E, T+1, D)
    pred = np.stack(pred_trajs)
    diff = pred - true
    # Per-step position/velocity MSE, averaged over balls and episodes.
    per_ball_diff = diff.reshape(*diff.shape[:-1], n_balls, 4)
    pos_err = (per_ball_diff[..., :2] ** 2).sum(-1).mean(axis=(0, 2))
    vel_err = (per_ball_diff[..., 2:] ** 2).sum(-1).mean(axis=(0, 2))
    return {
        "pos_mse_per_step": pos_err,
        "vel_mse_per_step": vel_err,
    }


def save_side_by_side(
    env_true: PhysicsEnv,
    env_pred: PhysicsEnv,
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    out_dir: Path,
    every: int = 4,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("pillow not installed; skipping image dump")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(0, true_traj.shape[0], every):
        _set_world_to_state(env_true, true_traj[t])
        _set_world_to_state(env_pred, pred_traj[t])
        env_true.renderer.draw(env_true.world)
        env_pred.renderer.draw(env_pred.world)
        left = env_true.renderer.frame()
        right = env_pred.renderer.frame()
        gap = np.full((left.shape[0], 4, 3), 60, dtype=np.uint8)
        side = np.concatenate([left, gap, right], axis=1)
        Image.fromarray(side).save(out_dir / f"cmp_{t:04d}.png")
    print(f"saved comparison frames -> {out_dir}/  (left=truth, right=model)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/world_model.pt"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="If set, dump side-by-side frames for one episode here.",
    )
    args = p.parse_args()

    model = load_model(args.ckpt, map_location=args.device)
    cfg_n_balls = model.state_dim // 4
    width = 320
    height = 240

    spec = SceneSpec(n_balls=cfg_n_balls, width=width, height=height)
    env = PhysicsEnv(width=width, height=height, headless=True)

    rng = np.random.default_rng(args.seed)
    true_all, pred_all = [], []
    for _ in range(args.episodes):
        t, p_ = run_episode(env, model, spec, args.horizon, rng)
        true_all.append(t)
        pred_all.append(p_)
    env.close()

    errs = horizon_errors(true_all, pred_all, cfg_n_balls)
    pos = errs["pos_mse_per_step"]
    vel = errs["vel_mse_per_step"]
    horizons = [1, 5, 10, 20, args.horizon]
    print(f"\nrollout errors over {args.episodes} episodes (per-ball MSE):")
    print("  horizon | pos_mse        vel_mse")
    for h in horizons:
        if h < len(pos):
            print(f"     t={h:3d} | {pos[h]:12.3f}  {vel[h]:12.3f}")

    if args.save_frames is not None:
        env_true = PhysicsEnv(width=width, height=height, headless=True)
        env_pred = PhysicsEnv(width=width, height=height, headless=True)
        rng2 = np.random.default_rng(args.seed + 1000)
        t, p_ = run_episode(env_true, model, spec, args.horizon, rng2)
        # ``run_episode`` advances env_true to the final state — that's fine,
        # we re-seed positions via ``_set_world_to_state`` per frame.
        # env_pred needs the same bodies (count/radius) but no real motion.
        _reset_scene(env_pred, spec, np.random.default_rng(args.seed + 1000))
        save_side_by_side(env_true, env_pred, t, p_, args.save_frames)
        env_true.close()
        env_pred.close()


if __name__ == "__main__":
    main()

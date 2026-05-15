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
from world_model.model import ballistic_step, reflect_walls, load as load_model


def _split_state(state: np.ndarray, n_balls: int) -> tuple[np.ndarray, np.ndarray]:
    s = state.reshape(n_balls, 4)
    return s[:, :2], s[:, 2:]


def _set_world_to_state(env: PhysicsEnv, state: np.ndarray) -> None:
    n = len(env.world.bodies)
    s = state.reshape(n, 4)
    for body, row in zip(env.world.bodies, s):
        body.pos = row[:2].astype(np.float64).copy()
        body.vel = row[2:].astype(np.float64).copy()


def baseline_rollout(
    s0: np.ndarray,
    horizon: int,
    dt: float,
    gravity_y: float,
    mass: float,
    wall_min: torch.Tensor | None = None,
    wall_max: torch.Tensor | None = None,
    wall_e: float = 0.0,
) -> np.ndarray:
    """Autoregressive analytic rollout (ballistic + optional wall reflection)."""
    s = torch.as_tensor(s0, dtype=torch.float32).unsqueeze(0)
    a = torch.zeros((1, s0.shape[0] // 2), dtype=torch.float32)
    out = [s.squeeze(0).numpy()]
    dt_t = torch.tensor(dt)
    g_t = torch.tensor(gravity_y)
    m_t = torch.tensor(mass)
    use_walls = wall_min is not None and wall_max is not None
    for _ in range(horizon):
        s = ballistic_step(s, a, dt_t, g_t, m_t)
        if use_walls:
            s = reflect_walls(s, wall_min, wall_max, wall_e)
        out.append(s.squeeze(0).numpy())
    return np.stack(out)


def run_episode(
    env: PhysicsEnv,
    model,
    spec: SceneSpec,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (true_traj, pred_traj, baseline_traj), each (horizon+1, state_dim)."""
    _reset_scene(env, spec, rng)
    s0 = env.world.state_vector().astype(np.float32)

    true_traj = [s0.copy()]
    for _ in range(horizon):
        env.step(action=None)
        true_traj.append(env.world.state_vector().astype(np.float32))
    true_traj = np.stack(true_traj)

    actions = np.zeros((horizon, 2 * spec.n_balls), dtype=np.float32)
    pred_traj = model.rollout(s0, actions)
    walls_set = bool(torch.isfinite(model.wall_min).all())
    base_traj = baseline_rollout(
        s0,
        horizon,
        dt=float(model.dt.item()),
        gravity_y=float(model.gravity_y.item()),
        mass=float(model.mass.item()),
        wall_min=model.wall_min if walls_set else None,
        wall_max=model.wall_max if walls_set else None,
        wall_e=float(model.wall_e.item()),
    )
    return true_traj, pred_traj, base_traj


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


def save_comparison_gif(
    env_true: PhysicsEnv,
    env_base: PhysicsEnv,
    env_pred: PhysicsEnv,
    true_traj: np.ndarray,
    base_traj: np.ndarray,
    pred_traj: np.ndarray,
    out_path: Path,
    fps: int = 30,
) -> None:
    """Three-panel animated GIF: truth | analytic baseline | learned model."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("pillow not installed; skipping GIF")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    horizon = true_traj.shape[0]
    for t in range(horizon):
        _set_world_to_state(env_true, true_traj[t])
        _set_world_to_state(env_base, base_traj[t])
        _set_world_to_state(env_pred, pred_traj[t])
        env_true.renderer.draw(env_true.world)
        env_base.renderer.draw(env_base.world)
        env_pred.renderer.draw(env_pred.world)
        a = env_true.renderer.frame()
        b = env_base.renderer.frame()
        c = env_pred.renderer.frame()
        gap = np.full((a.shape[0], 4, 3), 60, dtype=np.uint8)
        panel = np.concatenate([a, gap, b, gap, c], axis=1)
        img = Image.fromarray(panel)
        draw = ImageDraw.Draw(img)
        w = a.shape[1]
        draw.text((4, 4), "truth", fill=(255, 255, 255))
        draw.text((w + 8, 4), "baseline", fill=(255, 255, 255))
        draw.text((2 * w + 12, 4), "model", fill=(255, 255, 255))
        draw.text((panel.shape[1] - 50, 4), f"t={t}", fill=(200, 200, 200))
        frames.append(img)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
    print(f"saved GIF -> {out_path}  ({len(frames)} frames @ {fps} fps)")


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
    true_all, pred_all, base_all = [], [], []
    for _ in range(args.episodes):
        t, p_, b_ = run_episode(env, model, spec, args.horizon, rng)
        true_all.append(t)
        pred_all.append(p_)
        base_all.append(b_)
    env.close()

    model_errs = horizon_errors(true_all, pred_all, cfg_n_balls)
    base_errs = horizon_errors(true_all, base_all, cfg_n_balls)
    horizons = [1, 5, 10, 20, args.horizon]
    print(f"\nrollout errors over {args.episodes} episodes (per-ball MSE):")
    print("  horizon |   baseline pos    learned pos |   baseline vel    learned vel")
    for h in horizons:
        if h < len(model_errs["pos_mse_per_step"]):
            print(
                f"     t={h:3d} | {base_errs['pos_mse_per_step'][h]:13.3f}  "
                f"{model_errs['pos_mse_per_step'][h]:13.3f} | "
                f"{base_errs['vel_mse_per_step'][h]:13.3f}  "
                f"{model_errs['vel_mse_per_step'][h]:13.3f}"
            )

    if args.save_frames is not None:
        env_true_v = PhysicsEnv(width=width, height=height, headless=True)
        env_base_v = PhysicsEnv(width=width, height=height, headless=True)
        env_pred_v = PhysicsEnv(width=width, height=height, headless=True)
        rng2 = np.random.default_rng(args.seed + 1000)
        t_traj, p_traj, b_traj = run_episode(env_true_v, model, spec, args.horizon, rng2)
        # The other two envs need bodies in place so _set_world_to_state can
        # write into them; physics state is then overwritten per frame.
        _reset_scene(env_base_v, spec, np.random.default_rng(args.seed + 1000))
        _reset_scene(env_pred_v, spec, np.random.default_rng(args.seed + 1000))
        out_path = args.save_frames
        if out_path.suffix != ".gif":
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / "comparison.gif"
        save_comparison_gif(env_true_v, env_base_v, env_pred_v, t_traj, b_traj, p_traj, out_path)
        env_true_v.close()
        env_base_v.close()
        env_pred_v.close()


if __name__ == "__main__":
    main()

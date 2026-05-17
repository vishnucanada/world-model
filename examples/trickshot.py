"""Trick-shot solver: differentiable planning through the learned world model.

Given a cue ball and a target ball, find the cue's initial velocity that drives
the target as close as possible to a goal point. The shot is optimized by
gradient descent through the world model (treating it as a differentiable
simulator), then replayed in the real simulator to test whether the model's
plan transfers.

Usage:
    python -m examples.trickshot
    python -m examples.trickshot --goal 280 40 --steps 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model.env import PhysicsEnv
from world_model.model import load as load_model


def _set_world_to_state(env: PhysicsEnv, state: np.ndarray) -> None:
    n = len(env.world.bodies)
    s = state.reshape(n, 4)
    for body, row in zip(env.world.bodies, s):
        body.pos = row[:2].astype(np.float64).copy()
        body.vel = row[2:].astype(np.float64).copy()


def save_trickshot_gif(
    env_model: PhysicsEnv,
    env_real: PhysicsEnv,
    model_traj: np.ndarray,
    real_traj: np.ndarray,
    goal: np.ndarray,
    out_path: Path,
    fps: int = 30,
) -> None:
    from PIL import Image, ImageDraw

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    horizon = model_traj.shape[0]
    for t in range(horizon):
        _set_world_to_state(env_model, model_traj[t])
        _set_world_to_state(env_real, real_traj[t])
        env_model.renderer.draw(env_model.world)
        env_real.renderer.draw(env_real.world)
        a = env_model.renderer.frame()
        b = env_real.renderer.frame()
        gap = np.full((a.shape[0], 4, 3), 60, dtype=np.uint8)
        panel = np.concatenate([a, gap, b], axis=1)
        img = Image.fromarray(panel)
        draw = ImageDraw.Draw(img)
        w = a.shape[1]
        gx, gy = int(goal[0]), int(goal[1])
        for off_x in (0, w + 4):
            draw.line(
                [(off_x + gx - 7, gy - 7), (off_x + gx + 7, gy + 7)],
                fill=(255, 220, 0), width=2,
            )
            draw.line(
                [(off_x + gx - 7, gy + 7), (off_x + gx + 7, gy - 7)],
                fill=(255, 220, 0), width=2,
            )
        draw.text((4, 4), "model's plan", fill=(255, 255, 255))
        draw.text((w + 8, 4), "real outcome", fill=(255, 255, 255))
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
    print(f"saved trick-shot GIF -> {out_path}  ({len(frames)} frames @ {fps} fps)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/world_model_graph.pt"))
    p.add_argument("--out", type=Path, default=Path("viz/trickshot.gif"))
    p.add_argument("--horizon", type=int, default=60, help="rollout length in steps")
    p.add_argument("--steps", type=int, default=300, help="optimizer iterations")
    p.add_argument("--lr", type=float, default=8.0)
    p.add_argument("--init-speed", type=float, default=200.0)
    p.add_argument("--max-speed", type=float, default=250.0,
                   help="clamp cue speed after each opt step (training data had |v|<=212)")
    p.add_argument("--cue", type=float, nargs=2, default=[60.0, 120.0],
                   help="cue ball (x, y)")
    p.add_argument("--target", type=float, nargs=2, default=[180.0, 130.0],
                   help="target ball (x, y)")
    p.add_argument("--goal", type=float, nargs=2, default=[280.0, 40.0],
                   help="goal point (x, y)")
    args = p.parse_args()

    model = load_model(args.ckpt)
    for prm in model.parameters():
        prm.requires_grad_(False)
    cfg = torch.load(args.ckpt, weights_only=False)["config"]
    W, H = int(cfg["width"]), int(cfg["height"])

    cue_pos = np.array(args.cue, dtype=np.float32)
    target_pos = np.array(args.target, dtype=np.float32)
    goal = np.array(args.goal, dtype=np.float32)

    # Graph net is permutation-equivariant — runs cleanly at N=2 even though
    # it was trained at N=5. Per-ball stats apply uniformly.
    n_balls = 2
    scene = [
        (float(cue_pos[0]), float(cue_pos[1]), 0.0, 0.0),
        (float(target_pos[0]), float(target_pos[1]), 0.0, 0.0),
    ]

    init_state = np.array([v for ball in scene for v in ball], dtype=np.float32)
    base_state = torch.tensor(init_state)

    # Initialize cue velocity aimed at the *contact point* on the target ball
    # such that the line-of-centers at impact points toward the goal. Without
    # this prior, the optimizer easily falls into a "miss the target" local
    # minimum where the target stays put at its initial distance to the goal.
    goal_dir = (goal - target_pos)
    goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
    contact_point = target_pos - 2 * 12.0 * goal_dir  # 2 * ball radius
    to_contact = contact_point - cue_pos
    init_dir = to_contact / (np.linalg.norm(to_contact) + 1e-6)
    cue_v = torch.tensor(init_dir * args.init_speed, dtype=torch.float32, requires_grad=True)

    goal_t = torch.tensor(goal, dtype=torch.float32)
    zero_actions = torch.zeros(1, 2 * n_balls)

    opt = torch.optim.Adam([cue_v], lr=args.lr)
    print(f"optimizing cue velocity for {args.steps} steps, horizon={args.horizon}, lr={args.lr}")
    print(f"  init v=[{cue_v[0].item():+7.1f}, {cue_v[1].item():+7.1f}]")

    for it in range(args.steps):
        # Build initial state with cue velocity = optimization variable.
        s = torch.cat([base_state[:2], cue_v, base_state[4:]]).unsqueeze(0)
        dists2 = []
        cue_target_d2 = []
        for _ in range(args.horizon):
            s = model(s, zero_actions)
            ball_xy = s.reshape(1, n_balls, 4)[0, :, :2]
            target_xy = ball_xy[1]
            cue_xy = ball_xy[0]
            dists2.append(((target_xy - goal_t) ** 2).sum())
            cue_target_d2.append(((cue_xy - target_xy) ** 2).sum())
        d2_stack = torch.stack(dists2)
        # Primary loss: closest-approach of target to goal.
        # Mean term keeps pressure on if the optimizer drifts toward "miss".
        # Cue-target proximity ensures we actually hit (small weight; only matters
        # when target stays still and `min` is uninformative).
        actual_min_d2 = d2_stack.detach().min()
        loss = (
            d2_stack.min()
            + 0.05 * d2_stack.mean()
            + 0.1 * torch.stack(cue_target_d2).min()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            spd = cue_v.norm()
            if spd > args.max_speed:
                cue_v.mul_(args.max_speed / spd)
        if (it + 1) % 50 == 0 or it == 0:
            print(
                f"  iter {it+1:4d}  min_d={actual_min_d2.item()**0.5:6.1f}px  "
                f"loss={loss.item():8.1f}  "
                f"v=[{cue_v[0].item():+7.1f}, {cue_v[1].item():+7.1f}]"
            )

    final_v = cue_v.detach().numpy()
    print(
        f"\nfinal cue velocity: [{final_v[0]:+.1f}, {final_v[1]:+.1f}]  "
        f"(speed {np.linalg.norm(final_v):.1f} px/s)"
    )

    # Replay the optimized shot.
    init_state_with_v = init_state.copy()
    init_state_with_v[2:4] = final_v
    actions_np = np.zeros((args.horizon, 2 * n_balls), dtype=np.float32)
    model_traj = model.rollout(init_state_with_v, actions_np)

    env_real = PhysicsEnv(width=W, height=H, headless=True)
    for x, y, vx, vy in scene:
        env_real.add_ball(x=x, y=y, vx=vx, vy=vy, radius=12.0)
    env_real.world.bodies[0].vel = np.array(final_v, dtype=np.float64)
    real_traj = [env_real.world.state_vector().astype(np.float32).copy()]
    for _ in range(args.horizon):
        env_real.step(action=None)
        real_traj.append(env_real.world.state_vector().astype(np.float32).copy())
    real_traj = np.stack(real_traj)

    # Closest-approach distance for the target ball in each rollout.
    model_d = np.linalg.norm(model_traj[:, 4:6] - goal, axis=1)
    real_d = np.linalg.norm(real_traj[:, 4:6] - goal, axis=1)
    print("\nclosest approach of target ball to goal:")
    print(f"  model rollout: {model_d.min():6.1f} px at t={int(model_d.argmin())}")
    print(f"  real rollout : {real_d.min():6.1f} px at t={int(real_d.argmin())}")

    env_model_v = PhysicsEnv(width=W, height=H, headless=True)
    env_real_v = PhysicsEnv(width=W, height=H, headless=True)
    for env in (env_model_v, env_real_v):
        for x, y, vx, vy in scene:
            env.add_ball(x=x, y=y, vx=vx, vy=vy, radius=12.0)
    save_trickshot_gif(env_model_v, env_real_v, model_traj, real_traj, goal, args.out)
    env_model_v.close()
    env_real_v.close()
    env_real.close()


if __name__ == "__main__":
    main()

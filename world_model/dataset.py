"""Rollout dataset generator for the world model.

Drives the simulator with random initial conditions (and optional random
forces) and saves ``(state, action, next_state)`` transitions to a single
``.npz`` file along with normalization statistics.

The scene shape is fixed (N balls in a rectangular box) so the model has
a fixed input/output dimension.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import PhysicsEnv


@dataclass
class SceneSpec:
    n_balls: int = 5
    width: int = 320
    height: int = 240
    radius: float = 12.0
    mass: float = 1.0
    restitution: float = 0.85


def _reset_scene(env: PhysicsEnv, spec: SceneSpec, rng: np.random.Generator) -> None:
    env.world.bodies.clear()
    margin = spec.radius + 4.0
    for _ in range(spec.n_balls):
        env.add_ball(
            x=float(rng.uniform(margin, spec.width - margin)),
            y=float(rng.uniform(margin, spec.height - margin)),
            vx=float(rng.uniform(-150.0, 150.0)),
            vy=float(rng.uniform(-150.0, 150.0)),
            radius=spec.radius,
            mass=spec.mass,
            restitution=spec.restitution,
        )


def generate(
    out_path: str | Path,
    n_episodes: int = 200,
    steps_per_episode: int = 120,
    spec: SceneSpec | None = None,
    force_scale: float = 0.0,
    seed: int = 0,
) -> dict:
    """Generate transitions and write them to ``out_path`` as .npz.

    ``force_scale`` > 0 injects random per-step forces so the model learns
    a controlled dynamics, not just passive ballistic motion.
    """
    spec = spec or SceneSpec()
    rng = np.random.default_rng(seed)

    # Headless env — we don't need frames here, only state transitions.
    env = PhysicsEnv(
        width=spec.width,
        height=spec.height,
        headless=True,
    )

    state_dim = 4 * spec.n_balls
    action_dim = 2 * spec.n_balls
    total = n_episodes * steps_per_episode

    states = np.empty((total, state_dim), dtype=np.float32)
    actions = np.empty((total, action_dim), dtype=np.float32)
    next_states = np.empty((total, state_dim), dtype=np.float32)
    episode_ids = np.empty((total,), dtype=np.int32)

    idx = 0
    for ep in range(n_episodes):
        _reset_scene(env, spec, rng)
        s = env.world.state_vector().astype(np.float32)
        for _ in range(steps_per_episode):
            if force_scale > 0.0:
                a = rng.normal(0.0, force_scale, size=(spec.n_balls, 2)).astype(np.float32)
            else:
                a = np.zeros((spec.n_balls, 2), dtype=np.float32)
            env.step(action=a)
            s_next = env.world.state_vector().astype(np.float32)

            states[idx] = s
            actions[idx] = a.reshape(-1)
            next_states[idx] = s_next
            episode_ids[idx] = ep
            idx += 1
            s = s_next

    env.close()

    # Stats are computed on the inputs the model will actually see.
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-6
    delta = next_states - states
    delta_mean = delta.mean(axis=0)
    delta_std = delta.std(axis=0) + 1e-6

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=episode_ids,
        state_mean=state_mean,
        state_std=state_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        n_balls=np.int32(spec.n_balls),
        width=np.int32(spec.width),
        height=np.int32(spec.height),
        dt=np.float32(env.dt),
    )
    return {
        "transitions": int(total),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "path": str(out_path),
    }


if __name__ == "__main__":
    info = generate("data/transitions.npz", n_episodes=200, steps_per_episode=120)
    print(info)

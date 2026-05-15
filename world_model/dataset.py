"""Rollout dataset generator for the world model.

Saves full episode trajectories so the trainer can sample k-step
windows for multi-step rollout loss, along with the physics constants
the model needs to compute its analytic baseline.
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
    env.world.time = 0.0
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
    """Generate episode trajectories and write them to ``out_path`` as .npz."""
    spec = spec or SceneSpec()
    rng = np.random.default_rng(seed)
    env = PhysicsEnv(width=spec.width, height=spec.height, headless=True)

    state_dim = 4 * spec.n_balls
    action_dim = 2 * spec.n_balls

    states = np.empty((n_episodes, steps_per_episode + 1, state_dim), dtype=np.float32)
    actions = np.empty((n_episodes, steps_per_episode, action_dim), dtype=np.float32)

    for ep in range(n_episodes):
        _reset_scene(env, spec, rng)
        states[ep, 0] = env.world.state_vector()
        for t in range(steps_per_episode):
            if force_scale > 0.0:
                a = rng.normal(0.0, force_scale, size=(spec.n_balls, 2)).astype(np.float32)
            else:
                a = np.zeros((spec.n_balls, 2), dtype=np.float32)
            env.step(action=a)
            states[ep, t + 1] = env.world.state_vector()
            actions[ep, t] = a.reshape(-1)

    env.close()

    flat = states.reshape(-1, state_dim)
    state_mean = flat.mean(axis=0)
    state_std = flat.std(axis=0) + 1e-6

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=states,
        actions=actions,
        state_mean=state_mean,
        state_std=state_std,
        n_balls=np.int32(spec.n_balls),
        width=np.int32(spec.width),
        height=np.int32(spec.height),
        dt=np.float32(env.dt),
        gravity_y=np.float32(env.world.gravity[1]),
        mass=np.float32(spec.mass),
    )
    return {
        "episodes": n_episodes,
        "steps_per_episode": steps_per_episode,
        "transitions": n_episodes * steps_per_episode,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "path": str(out_path),
    }


if __name__ == "__main__":
    info = generate("data/transitions.npz", n_episodes=200, steps_per_episode=120)
    print(info)

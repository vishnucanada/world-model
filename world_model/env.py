"""Gym-style wrapper bundling physics + rendering for world-model training.

Each step() returns a dict with both modalities:
  - "state": flat float64 numpy vector of body positions/velocities
  - "frame": (H, W, 3) uint8 RGB image
"""
from __future__ import annotations

import numpy as np

from .physics import Body, Wall, World, vec2
from .render import Renderer


class PhysicsEnv:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        dt: float = 1.0 / 60.0,
        substeps: int = 2,
        gravity=(0.0, 900.0),
        headless: bool = True,
    ):
        self.width = width
        self.height = height
        self.dt = dt
        self.substeps = substeps
        self.world = World(gravity=vec2(*gravity))
        self.renderer = Renderer(width, height, headless=headless)
        self._build_default_scene()

    def _build_default_scene(self) -> None:
        m = 8
        w, h = self.width, self.height
        self.world.add_wall(Wall(vec2(m, m), vec2(w - m, m)))
        self.world.add_wall(Wall(vec2(w - m, m), vec2(w - m, h - m)))
        self.world.add_wall(Wall(vec2(w - m, h - m), vec2(m, h - m)))
        self.world.add_wall(Wall(vec2(m, h - m), vec2(m, m)))

    def add_ball(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        radius: float = 18.0,
        mass: float = 1.0,
        restitution: float = 0.85,
        color: tuple = (220, 90, 90),
    ) -> Body:
        return self.world.add_body(
            Body(
                pos=vec2(x, y),
                vel=vec2(vx, vy),
                radius=radius,
                mass=mass,
                restitution=restitution,
                color=color,
            )
        )

    def step(self, action=None) -> dict:
        """``action`` is an (N, 2) array of forces, one per body, or None."""
        if action is not None:
            action = np.asarray(action, dtype=np.float64)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            # Re-apply per substep: world.step zeros _force after integrating,
            # so without this the action would only act during the first substep.
            if action is not None:
                for body, force in zip(self.world.bodies, action):
                    body.apply_force(force)
            self.world.step(h)
        self.renderer.draw(self.world)
        return self.observation()

    def observation(self) -> dict:
        return {
            "state": self.world.state_vector(),
            "frame": self.renderer.frame(),
            "time": self.world.time,
        }

    def render(self, fps: int = 60) -> None:
        self.renderer.flip(fps=fps)

    def close(self) -> None:
        self.renderer.close()

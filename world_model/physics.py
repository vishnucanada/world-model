"""2D rigid-body physics: circles + static line-segment walls.

Designed for use as a world-model environment: deterministic step(),
no hidden time, easy state introspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Vec2 = np.ndarray


def vec2(x: float, y: float) -> Vec2:
    return np.array([x, y], dtype=np.float64)


@dataclass
class Body:
    pos: Vec2
    vel: Vec2
    radius: float
    mass: float = 1.0
    restitution: float = 0.85
    linear_damping: float = 0.0
    color: tuple = (220, 220, 220)
    _force: Vec2 = field(default_factory=lambda: vec2(0.0, 0.0))

    @property
    def inv_mass(self) -> float:
        return 0.0 if self.mass == float("inf") else 1.0 / self.mass

    def apply_force(self, f) -> None:
        self._force = self._force + np.asarray(f, dtype=np.float64)

    def apply_impulse(self, j) -> None:
        self.vel = self.vel + self.inv_mass * np.asarray(j, dtype=np.float64)


@dataclass
class Wall:
    """Static line segment from ``a`` to ``b``."""
    a: Vec2
    b: Vec2
    restitution: float = 0.9
    color: tuple = (110, 115, 135)


def _closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2:
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return a
    t = float((p - a) @ ab) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return a + t * ab


class World:
    def __init__(self, gravity: Vec2 = None):
        self.gravity = vec2(0.0, 900.0) if gravity is None else np.asarray(gravity, dtype=np.float64)
        self.bodies: list[Body] = []
        self.walls: list[Wall] = []
        self.time: float = 0.0

    def add_body(self, body: Body) -> Body:
        self.bodies.append(body)
        return body

    def add_wall(self, wall: Wall) -> Wall:
        self.walls.append(wall)
        return wall

    def step(self, dt: float) -> None:
        # semi-implicit Euler: integrate forces -> velocity, then velocity -> position.
        for b in self.bodies:
            if b.inv_mass == 0.0:
                b._force = vec2(0.0, 0.0)
                continue
            accel = self.gravity + b._force * b.inv_mass
            b.vel = b.vel + accel * dt
            if b.linear_damping:
                b.vel = b.vel * max(0.0, 1.0 - b.linear_damping * dt)
            b._force = vec2(0.0, 0.0)

        for b in self.bodies:
            b.pos = b.pos + b.vel * dt

        self._resolve_circle_circle()
        self._resolve_circle_wall()

        self.time += dt

    def _resolve_circle_circle(self) -> None:
        n = len(self.bodies)
        for i in range(n):
            a = self.bodies[i]
            for j in range(i + 1, n):
                b = self.bodies[j]
                inv_sum = a.inv_mass + b.inv_mass
                if inv_sum == 0.0:
                    continue
                delta = b.pos - a.pos
                dist2 = float(delta @ delta)
                r = a.radius + b.radius
                if dist2 >= r * r or dist2 == 0.0:
                    continue
                dist = dist2 ** 0.5
                normal = delta / dist
                penetration = r - dist
                correction = (penetration / inv_sum) * normal
                a.pos = a.pos - correction * a.inv_mass
                b.pos = b.pos + correction * b.inv_mass
                rv = b.vel - a.vel
                vn = float(rv @ normal)
                if vn > 0:
                    continue
                e = min(a.restitution, b.restitution)
                jmag = -(1.0 + e) * vn / inv_sum
                impulse = jmag * normal
                a.vel = a.vel - impulse * a.inv_mass
                b.vel = b.vel + impulse * b.inv_mass

    def _resolve_circle_wall(self) -> None:
        for body in self.bodies:
            if body.inv_mass == 0.0:
                continue
            for wall in self.walls:
                cp = _closest_point_on_segment(body.pos, wall.a, wall.b)
                delta = body.pos - cp
                dist2 = float(delta @ delta)
                if dist2 >= body.radius * body.radius or dist2 == 0.0:
                    continue
                dist = dist2 ** 0.5
                normal = delta / dist
                penetration = body.radius - dist
                body.pos = body.pos + normal * penetration
                vn = float(body.vel @ normal)
                if vn >= 0:
                    continue
                e = min(body.restitution, wall.restitution)
                body.vel = body.vel - (1.0 + e) * vn * normal

    def state_vector(self) -> np.ndarray:
        """Flat per-body [x, y, vx, vy] vector. Useful as model input."""
        if not self.bodies:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate([np.concatenate([b.pos, b.vel]) for b in self.bodies])

    def state_dict(self) -> dict:
        return {
            "time": self.time,
            "bodies": [
                {"pos": b.pos.tolist(), "vel": b.vel.tolist(), "radius": b.radius, "mass": b.mass}
                for b in self.bodies
            ],
            "walls": [{"a": w.a.tolist(), "b": w.b.tolist()} for w in self.walls],
        }

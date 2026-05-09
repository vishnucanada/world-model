"""Pygame renderer. Returns frames as (H, W, 3) uint8 RGB arrays."""
from __future__ import annotations

import os

import numpy as np

from .physics import World


class Renderer:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        bg: tuple = (20, 22, 30),
        headless: bool = False,
    ):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # imported lazily so headless callers without a display can opt in first

        self._pygame = pygame
        pygame.init()
        self.width = width
        self.height = height
        self.bg = bg
        self.headless = headless
        flags = 0 if headless else pygame.SHOWN
        self.surface = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("world-model physics")
        self.clock = pygame.time.Clock()

    def draw(self, world: World) -> None:
        pg = self._pygame
        self.surface.fill(self.bg)
        for wall in world.walls:
            pg.draw.line(
                self.surface,
                wall.color,
                wall.a.astype(int).tolist(),
                wall.b.astype(int).tolist(),
                2,
            )
        for body in world.bodies:
            pg.draw.circle(
                self.surface,
                body.color,
                body.pos.astype(int).tolist(),
                int(body.radius),
            )

    def flip(self, fps: int = 60) -> None:
        self._pygame.display.flip()
        self.clock.tick(fps)

    def frame(self) -> np.ndarray:
        """Current surface as an (H, W, 3) uint8 RGB array."""
        arr = self._pygame.surfarray.array3d(self.surface)  # (W, H, 3)
        return np.transpose(arr, (1, 0, 2))

    def close(self) -> None:
        self._pygame.quit()

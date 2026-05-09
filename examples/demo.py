"""Interactive demo: click to drop balls, R to reset, ESC to quit."""
import random
import sys

import pygame

from world_model import PhysicsEnv, Wall, vec2


def main() -> None:
    env = PhysicsEnv(width=800, height=600, headless=False)
    env.world.add_wall(Wall(vec2(100, 250), vec2(420, 380)))
    env.world.add_wall(Wall(vec2(700, 280), vec2(450, 460)))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.world.bodies.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                color = (
                    random.randint(120, 255),
                    random.randint(120, 255),
                    random.randint(120, 255),
                )
                env.add_ball(
                    x, y,
                    radius=random.uniform(10, 25),
                    color=color,
                )

        env.step()
        env.render(fps=60)

    env.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

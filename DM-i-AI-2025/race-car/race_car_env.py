import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from src.game.core import (
    initialize_game_state,
    update_game,
    MAX_TICKS,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)
from src.game import core


class RaceCarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(5)  # Corresponds to 5 actions
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(16,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        initialize_game_state(api_url="", seed_value=seed)
        self.state = core.STATE

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        action_map = {
            0: "ACCELERATE",
            1: "DECELERATE",
            2: "STEER_LEFT",
            3: "STEER_RIGHT",
            4: "NOTHING",
        }

        self.state = update_game(action_map[action])

        terminated = self.state.crashed or self.state.ticks > MAX_TICKS
        reward = self.state.distance
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))  # Clear the screen with black
        self.screen.blit(self.state.road.surface, (0, 0))

        for wall in self.state.road.walls:
            wall.draw(self.screen)

        for car in self.state.cars:
            if car.sprite:
                self.screen.blit(car.sprite, (car.x, car.y))
            else:
                pygame.draw.rect(self.screen, (255, 0, 0), car.rect)

        if self.state.sensors_enabled:
            for sensor in self.state.sensors:
                sensor.draw(self.screen)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        return np.array(
            [
                sensor.reading
                if sensor.reading is not None
                else self.observation_space.high[0]
                for sensor in self.state.sensors
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {"distance": self.state.distance, "ticks": self.state.ticks}

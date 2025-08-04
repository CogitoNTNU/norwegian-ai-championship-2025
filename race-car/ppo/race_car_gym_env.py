import gymnasium as gym
import numpy as np
import pygame
from typing import Optional, Dict, Any, Tuple
import sys
import os

# Add parent directory to path to import game modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.core import (
    initialize_game_state,
    update_game,
    intersects,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    MAX_TICKS,
)
import src.game.core as game_core


class RaceCarEnv(gym.Env):
    """Clean Gym environment for the race car game."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode
        self.seed_value = seed or "default_seed"

        # Action space: 0=NOTHING, 1=ACCELERATE, 2=DECELERATE (no steering)
        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {
            0: "NOTHING",
            1: "ACCELERATE",
            2: "DECELERATE",
        }

        # Initialize pygame first (required for game state)
        pygame.init()

        # We need to initialize game state first to know sensor count
        initialize_game_state("", self.seed_value, sensor_removal=0)
        self.num_sensors = len(game_core.STATE.sensors)

        # Observation space: sensor readings + velocity + lane position
        # Sensors: num_sensors values (0-1000, normalized to 0-1)
        # Velocity: x and y components (normalized)
        # Lane position: y position relative to road (0-1)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.num_sensors + 3,), dtype=np.float32
        )

        # Initialize display if rendering
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Race Car PPO Training (No Steering)")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        # Training metrics
        self.episode_reward = 0
        self.steps_without_progress = 0
        self.max_steps_without_progress = 100
        self.last_reward_breakdown = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize game state
        initialize_game_state("", self.seed_value, sensor_removal=0)

        # Reset metrics
        self.episode_reward = 0
        self.steps_without_progress = 0
        self.last_reward_breakdown = {}

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Convert action to string
        action_str = self.action_map[action]

        # Store previous state
        prev_distance = game_core.STATE.distance
        prev_velocity_x = game_core.STATE.ego.velocity.x

        # Execute action
        update_game(action_str)

        # Increment ticks (update_game doesn't do this)
        game_core.STATE.ticks += 1

        # Check collisions (update_game doesn't do this)
        for car in game_core.STATE.cars:
            if car != game_core.STATE.ego and intersects(
                game_core.STATE.ego.rect, car.rect
            ):
                game_core.STATE.crashed = True

        # Check collision with walls
        for wall in game_core.STATE.road.walls:
            if intersects(game_core.STATE.ego.rect, wall.rect):
                game_core.STATE.crashed = True

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(prev_distance, prev_velocity_x)
        self.episode_reward += reward

        # Check termination conditions
        terminated = game_core.STATE.crashed or game_core.STATE.ticks >= MAX_TICKS
        truncated = self.steps_without_progress >= self.max_steps_without_progress

        # Get info
        info = self._get_info()

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation from game state."""
        obs = []

        # 1. Sensor readings (16 values, normalized to 0-1)
        for sensor in game_core.STATE.sensors:
            if sensor.reading is not None:
                # Normalize sensor reading (0-1000 -> 0-1)
                normalized_reading = min(sensor.reading / 1000.0, 1.0)
            else:
                # No reading means maximum distance
                normalized_reading = 1.0
            obs.append(normalized_reading)

        # 2. Velocity (normalized)
        # Typical max velocity is around 20-30
        vel_x_normalized = np.clip(game_core.STATE.ego.velocity.x / 30.0, 0.0, 1.0)
        vel_y_normalized = np.clip(
            (game_core.STATE.ego.velocity.y) / 10.0, -1.0, 1.0
        )  # y can be negative
        obs.extend([vel_x_normalized, vel_y_normalized])

        # 3. Lane position (normalized y position)
        y_position = (game_core.STATE.ego.y - game_core.STATE.road.y_start) / (
            game_core.STATE.road.y_end - game_core.STATE.road.y_start
        )
        y_position = np.clip(y_position, 0.0, 1.0)
        obs.append(y_position)

        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, prev_distance: float, prev_velocity_x: float) -> float:
        """Calculate reward based on game state changes."""
        reward = 0.0
        reward_breakdown = {}

        # 1. Distance reward (main objective)
        distance_gained = game_core.STATE.distance - prev_distance
        distance_reward = distance_gained * 0.1  # Scale down to reasonable values
        reward += distance_reward
        reward_breakdown["distance_reward"] = distance_reward

        # 4. Collision penalty
        collision_penalty = 0.0
        if game_core.STATE.crashed:
            collision_penalty = -100.0  # Large penalty for crashing
        reward += collision_penalty
        reward_breakdown["collision_penalty"] = collision_penalty

        self.last_reward_breakdown = reward_breakdown

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        info = {
            "distance": game_core.STATE.distance,
            "velocity_x": game_core.STATE.ego.velocity.x,
            "velocity_y": game_core.STATE.ego.velocity.y,
            "crashed": game_core.STATE.crashed,
            "ticks": game_core.STATE.ticks,
            "episode_reward": self.episode_reward,
            "y_position": game_core.STATE.ego.y,
            "reward_breakdown": self.last_reward_breakdown.copy(),
        }

        # Add episode info when episode ends
        if (
            game_core.STATE.crashed
            or game_core.STATE.ticks >= MAX_TICKS
            or self.steps_without_progress >= self.max_steps_without_progress
        ):
            info["episode"] = {
                "r": self.episode_reward,
                "l": game_core.STATE.ticks,
                "distance": game_core.STATE.distance,
            }

        return info

    def render(self):
        """Render the game state."""
        if self.render_mode == "human" and self.screen is not None:
            # Clear screen
            self.screen.fill((0, 0, 0))

            # Draw road
            self.screen.blit(game_core.STATE.road.surface, (0, 0))

            # Draw walls
            for wall in game_core.STATE.road.walls:
                wall.draw(self.screen)

            # Draw cars
            for car in game_core.STATE.cars:
                if car.sprite:
                    self.screen.blit(car.sprite, (car.x, car.y))
                    # Draw bounding box
                    bounds = car.get_bounds()
                    color = (255, 0, 0) if car == game_core.STATE.ego else (0, 255, 0)
                    pygame.draw.rect(self.screen, color, bounds, width=2)

            # Draw sensors
            if game_core.STATE.sensors_enabled:
                for sensor in game_core.STATE.sensors:
                    sensor.draw(self.screen)

            # Draw info text
            font = pygame.font.Font(None, 36)
            info_texts = [
                f"Distance: {game_core.STATE.distance:.1f}",
                f"Velocity: {game_core.STATE.ego.velocity.x:.1f}",
                f"Reward: {self.episode_reward:.1f}",
                f"Tick: {game_core.STATE.ticks}",
                "NO STEERING MODE",
            ]

            y_offset = 10
            for text in info_texts:
                text_surface = font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 40

            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()


# Test the environment
if __name__ == "__main__":
    env = RaceCarEnv(render_mode="human")
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print("Actions: 0=NOTHING, 1=ACCELERATE, 2=DECELERATE (No steering)")

    # Run a few random steps
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                f"Episode ended - Distance: {info['distance']}, Reward: {env.episode_reward}"
            )
            obs, info = env.reset()

    env.close()

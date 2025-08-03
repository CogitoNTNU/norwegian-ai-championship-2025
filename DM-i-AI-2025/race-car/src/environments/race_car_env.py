import numpy as np
import gymnasium as gym
from gymnasium import spaces
import src.game.core as core
from src.game.core import initialize_game_state, update_game, intersects

# Try to import pygame, but make it optional for headless training
try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError as e:
    PYGAME_AVAILABLE = False
    print(f"Warning: pygame not available: {e}")
    print("Running in headless mode only")


class RealRaceCarEnv(gym.Env):
    """
    PPO environment that uses the actual race car game logic.
    Supports 3-game batches with 1-minute games.
    Crashes end episodes immediately.
    """

    def __init__(self, seed_value=None, headless=True):
        super().__init__()
        self.seed_value = seed_value
        self.headless = headless
        self.max_steps_per_game = 3600  # 60 seconds at 60 FPS per game
        self.games_per_batch = 1
        self.current_step = 0
        self.current_game = 0
        self.crashed_steps = 0
        self.max_crashed_steps = 0  # End game instantly on crash

        # Batch tracking
        self.batch_rewards = []
        self.batch_distances = []

        # Initialize pygame only if available and not headless
        if not self.headless and not PYGAME_AVAILABLE:
            print("Warning: pygame not available, forcing headless mode")
            self.headless = True

        if not self.headless and PYGAME_AVAILABLE:
            pygame.init()

        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )
        self.action_map = {
            0: "NOTHING",
            1: "ACCELERATE",
            2: "DECELERATE",
            3: "STEER_LEFT",
            4: "STEER_RIGHT",
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed_value = seed

        self.current_game = 0
        self.batch_rewards = []
        self.batch_distances = []
        self._reset_single_game()
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _reset_single_game(self):
        initialize_game_state("dummy_url", self.seed_value)
        core.STATE.cars = [core.STATE.ego]
        self.current_step = 0
        self.crashed_steps = 0
        self._last_distance = core.STATE.distance
        self._following_steps = 0
        self._last_y = core.STATE.ego.y

    def step(self, action):
        action_idx = int(action) if hasattr(action, "__iter__") else action
        action_str = self.action_map[action_idx]
        crashed_before = core.STATE.crashed

        if core.STATE.crashed:
            self.crashed_steps += 1
        else:
            update_game(action_str)
            # Check collisions after game update, similar to _working version
            self._check_collisions()

        self.current_step += 1

        obs = self._get_observation()
        reward = self._calculate_reward(crashed_before)

        # --- Game ends instantly on crash or after timeout ---
        is_crash = core.STATE.crashed and self.crashed_steps >= self.max_crashed_steps
        is_timeout = self.current_step >= self.max_steps_per_game

        terminated = bool(is_crash)  # Ended by crash (failure)
        truncated = bool(
            is_timeout and not core.STATE.crashed
        )  # Ended by time (success if no crash)

        game_finished = terminated or truncated

        if game_finished:
            self.batch_rewards.append(reward)
            self.batch_distances.append(core.STATE.distance)

            self.current_game += 1

            if not (self.current_game >= self.games_per_batch):
                self._reset_single_game()
                terminated = False
                truncated = False
        else:
            terminated = False
            truncated = False

        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        sensor_readings = []
        for sensor in core.STATE.sensors:
            # Use inverse normalization: close objects = higher values (more urgent)
            # This makes it easier for the model to learn that high sensor values = danger
            if sensor.reading is not None:
                # Inverse normalize: 0=far/safe, 1=close/danger
                # This way close objects (50) become 0.95, far objects (1000) become 0.0
                normalized_distance = sensor.reading / 1000.0
            else:
                # No reading = nothing detected = safe = 0
                normalized_distance = 1.0
            sensor_readings.append(normalized_distance)

        return np.array(sensor_readings, dtype=np.float32)

    def _check_collisions(self):
        # Handle car collisions
        for car in core.STATE.cars:
            if car != core.STATE.ego and intersects(core.STATE.ego.rect, car.rect):
                core.STATE.crashed = True
                return

        # Handle wall collisions with improved logic from _working version
        # Only crash if the car goes significantly beyond the wall boundary
        for wall in core.STATE.road.walls:
            if intersects(core.STATE.ego.rect, wall.rect):
                # Calculate overlap distance
                x_overlap = min(core.STATE.ego.rect.right, wall.rect.right) - max(
                    core.STATE.ego.rect.left, wall.rect.left
                )
                y_overlap = min(core.STATE.ego.rect.bottom, wall.rect.bottom) - max(
                    core.STATE.ego.rect.top, wall.rect.top
                )

                # Only crash if there's significant overlap (more than just touching)
                overlap_threshold = 10  # pixels
                if x_overlap > overlap_threshold and y_overlap > overlap_threshold:
                    core.STATE.crashed = True
                    return

    def _calculate_reward(self, crashed_before):
        speed = core.STATE.ego.velocity.x
        if speed < 5:
            speed_reward = -0.1
        else:
            speed_reward = speed / 20.0

        # Distance progress reward: reward when car makes forward progress
        distance_progress = core.STATE.distance - self._last_distance
        distance_reward = (
            max(0, distance_progress) * 0.1
        )  # Scale factor for distance progress
        self._last_distance = core.STATE.distance

        crash_penalty = -1000.0 if (core.STATE.crashed and not crashed_before) else 0.0
        completion_bonus = (
            1000.0
            if self.current_step >= self.max_steps_per_game and not core.STATE.crashed
            else 0.0
        )

        reward = speed_reward + distance_reward + crash_penalty + completion_bonus
        self._reward_breakdown = {
            "speed_reward": speed_reward,
            "distance_reward": distance_reward,
            "crash_penalty": crash_penalty,
            "completion_bonus": completion_bonus,
        }
        return reward

    def _get_info(self):
        info = {
            "distance": core.STATE.distance,
            "speed": core.STATE.ego.velocity.x,
            "crashed": core.STATE.crashed,
            "ticks": core.STATE.ticks,
            "cars_nearby": len([c for c in core.STATE.cars if c != core.STATE.ego]),
            "race_completed": self.current_step >= self.max_steps_per_game
            and not core.STATE.crashed,
            "time_remaining": max(0, self.max_steps_per_game - self.current_step) / 60,
            "current_game": self.current_game + 1,
            "games_per_batch": self.games_per_batch,
            "batch_distances": self.batch_distances.copy(),
            "batch_total_distance": sum(self.batch_distances)
            if self.batch_distances
            else 0,
            "reward_breakdown": getattr(self, "_reward_breakdown", {}).copy(),
        }
        return info

    def close(self):
        pass

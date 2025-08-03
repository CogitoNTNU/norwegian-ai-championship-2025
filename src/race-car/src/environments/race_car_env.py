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
        # Observation space: 16 sensors (0-1000) + 7 state features
        obs_low = np.zeros(23, dtype=np.float32)
        obs_high = np.full(23, 1000.0, dtype=np.float32)
        # State features have different ranges
        obs_high[16:] = 1.0  # Last 7 features are normalized 0-1
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
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
        self._last_cars_ahead = 0  # Initialize overtaking tracker

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

            if self.current_game >= self.games_per_batch:
                # Batch complete - let PPO update
                pass
            else:
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
            # Normalize sensor readings to 0-1 range
            if sensor.reading is not None:
                # Normalize: 0 = far/safe, 1 = close/danger
                normalized_reading = min(sensor.reading / 1000.0, 1.0)
            else:
                # No reading = nothing detected = max distance = safe
                normalized_reading = 1.0
            sensor_readings.append(normalized_reading)

        while len(sensor_readings) < 16:
            sensor_readings.append(1.0)  # Padding with 1.0 = max distance/safe
        sensor_readings = sensor_readings[:16]

        ego_car = core.STATE.ego

        # Additional useful state information
        velocity_x = np.clip(ego_car.velocity.x / 20.0, 0.0, 1.0)
        velocity_y = np.clip(
            (ego_car.velocity.y + 10.0) / 20.0, 0.0, 1.0
        )  # Normalized to 0-1
        position_y = ego_car.y / 1200.0

        # Calculate relative positions of nearby cars
        cars_ahead = 0
        cars_behind = 0
        closest_car_ahead_distance = 1.0
        closest_car_behind_distance = 1.0

        for car in core.STATE.cars:
            if car != core.STATE.ego:
                relative_x = car.x - ego_car.x
                if relative_x > 0:  # Car is ahead
                    cars_ahead += 1
                    closest_car_ahead_distance = min(
                        closest_car_ahead_distance, relative_x / 1000.0
                    )
                else:  # Car is behind
                    cars_behind += 1
                    closest_car_behind_distance = min(
                        closest_car_behind_distance, abs(relative_x) / 1000.0
                    )

        # Normalize car counts
        cars_ahead_normalized = min(cars_ahead / 5.0, 1.0)
        cars_behind_normalized = min(cars_behind / 5.0, 1.0)

        observation = np.array(
            sensor_readings
            + [
                velocity_x,
                velocity_y,
                position_y,
                cars_ahead_normalized,
                cars_behind_normalized,
                closest_car_ahead_distance,
                closest_car_behind_distance,
            ],
            dtype=np.float32,
        )
        return observation

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
        # Base speed reward - encourage good speed but not reckless
        # Scale: more generous to encourage movement
        speed = core.STATE.ego.velocity.x
        if speed < 3:  # Lowered threshold
            speed_reward = 0.0  # No penalty, just no reward
        elif speed > 18:
            speed_reward = 0.03  # Increased reward for good speed
        else:
            speed_reward = speed / 600.0  # More generous scaling

        # Overtaking reward - count cars that moved from ahead to behind
        # Give significant one-time bonus for overtaking
        overtaking_reward = 0.0
        if hasattr(self, "_last_cars_ahead"):
            cars_ahead_now = sum(
                1
                for car in core.STATE.cars
                if car != core.STATE.ego and car.x > core.STATE.ego.x
            )
            if cars_ahead_now < self._last_cars_ahead:
                # Successfully overtook a car!
                overtaking_reward = 5.0 * (self._last_cars_ahead - cars_ahead_now)
            self._last_cars_ahead = cars_ahead_now
        else:
            self._last_cars_ahead = sum(
                1
                for car in core.STATE.cars
                if car != core.STATE.ego and car.x > core.STATE.ego.x
            )

        # Distance progress reward - make this more significant
        distance_reward = 0.0
        if hasattr(self, "_last_distance"):
            distance_progress = core.STATE.distance - self._last_distance
            distance_reward = (
                distance_progress / 500.0
            )  # Increased to incentivize forward progress
        self._last_distance = core.STATE.distance

        # Survival bonus - reward staying alive each step
        survival_bonus = 0.01

        # Proximity penalty - discourage staying too close to other cars
        proximity_penalty = 0.0
        min_distance = float("inf")
        for sensor in core.STATE.sensors:
            if sensor.reading is not None and sensor.reading < min_distance:
                min_distance = sensor.reading

        if min_distance < 50:  # Very close
            proximity_penalty = -0.005
        elif min_distance < 100:  # Close
            proximity_penalty = -0.002

        # Remove steering penalty - we want the agent to steer for overtaking!

        # Crash and completion - reduce crash penalty for better exploration
        crash_penalty = -50.0 if (core.STATE.crashed and not crashed_before) else 0.0
        completion_bonus = (
            100.0
            if self.current_step >= self.max_steps_per_game and not core.STATE.crashed
            else 0.0
        )

        # Total reward - removed steering_penalty, added survival_bonus
        reward = (
            speed_reward
            + overtaking_reward
            + distance_reward
            + proximity_penalty
            + crash_penalty
            + completion_bonus
            + survival_bonus
        )

        self._reward_breakdown = {
            "speed_reward": speed_reward,
            "overtaking_reward": overtaking_reward,
            "distance_reward": distance_reward,
            "proximity_penalty": proximity_penalty,
            "crash_penalty": crash_penalty,
            "completion_bonus": completion_bonus,
            "survival_bonus": survival_bonus,
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

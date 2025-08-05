import numpy as np
import gymnasium as gym
from gymnasium import spaces
import src.game.core as core
from src.game.core import initialize_game_state, update_game, intersects

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
        # Observation space: 16 sensors + 4 additional values (x, y, velocity_x, velocity_y)
        obs_low = np.zeros(20, dtype=np.float32)
        obs_high = np.ones(20, dtype=np.float32)
        # Set higher bounds for position coordinates
        obs_high[16] = 2000.0  # x coordinate max
        obs_high[17] = 1200.0  # y coordinate max
        obs_high[18] = 30.0  # velocity_x max
        obs_high[19] = 20.0  # velocity_y max
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
        self._following_steps = 0
        self._last_y = core.STATE.ego.y
        self._last_cars_ahead = 0  # Initialize overtaking tracker
        self._last_distance = 0.0  # Track previous distance for reward calculation

        # Initialize accumulated reward tracking
        self._accumulated_rewards = {
            "distance_reward": 0.0,
            "crash_penalty": 0.0,
            "completion_bonus": 0.0,
        }

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

        # --- Game ends instantly on crash or after timeout ---
        is_crash = core.STATE.crashed and self.crashed_steps >= self.max_crashed_steps
        is_timeout = self.current_step >= self.max_steps_per_game

        terminated = bool(is_crash)  # Ended by crash (failure)
        truncated = bool(
            is_timeout and not core.STATE.crashed
        )  # Ended by time (success if no crash)

        game_finished = terminated or truncated

        obs = self._get_observation()
        reward = self._calculate_reward(crashed_before)

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

        # Add car position and velocity to observation
        additional_values = [
            ego_car.x / 20_000,  # x coordinate
            ego_car.y / 1_000,  # y coordinate
            ego_car.velocity.x / 15,  # x velocity
            ego_car.velocity.y / 15,  # y velocity
        ]

        observation = np.array(
            sensor_readings + additional_values,
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
        current_distance = core.STATE.distance
        distance_reward = (
            current_distance - self._last_distance
        ) / 10.0  # Reward based on distance progress
        self._last_distance = current_distance  # Update last distance for next step

        speed_penalty = 0.0
        proximity_penalty = 0.0
        front_sensor = core.STATE.sensors[4]
        if front_sensor.reading is not None:
            proximity_penalty -= (1 / front_sensor.reading) * 10_000

            speed = core.STATE.ego.velocity.x

            if speed > 10:  # Only penalize if going fast
                danger_factor = 1.0 - (
                    front_sensor.reading / 200.0
                )  # 0 to 1, where 1 is very close
                speed_penalty = -danger_factor * (speed - 10) * 0.5

        # Remove steering penalty - we want the agent to steer for overtaking!

        # Crash and completion - reduce crash penalty for better exploration
        crash_penalty = -50.0 if (core.STATE.crashed and not crashed_before) else 0.0
        completion_bonus = (
            100.0
            if self.current_step >= self.max_steps_per_game and not core.STATE.crashed
            else 0.0
        )

        collision_risk_penalty = 0.0

        # Get velocity components
        vx = core.STATE.ego.velocity.x
        vy = core.STATE.ego.velocity.y

        # Check each sensor
        sensor_angles = [
            0,
            22.5,
            45,
            67.5,
            90,
            112.5,
            135,
            157.5,
            180,
            202.5,
            225,
            247.5,
            270,
            292.5,
            315,
            337.5,
        ]

        for i, sensor in enumerate(core.STATE.sensors[:16]):
            if (
                sensor.reading is not None and sensor.reading < 300
            ):  # Object detected within danger zone
                # Convert sensor angle to radians
                angle_rad = np.radians(sensor_angles[i])

                # Calculate closing rate (velocity component in direction of sensor)
                # For a car facing right (0°), sensor at 0° points left, 90° points forward, etc.
                # Adjust based on your coordinate system
                closing_rate = vx * np.cos(angle_rad) + vy * np.sin(angle_rad)

                # Only penalize if we're moving toward the object (positive closing rate)
                if closing_rate > 0:
                    # Scale penalty by both closing rate and proximity
                    proximity_factor = 1.0 - (
                        sensor.reading / 300.0
                    )  # 0 to 1, where 1 is very close

                    # Quadratic penalty for high closing rates when close
                    collision_risk_penalty -= closing_rate * proximity_factor**2 * 0.1

                    # Extra penalty for front sensors (indices 2-6) as they're most critical
                    if 2 <= i <= 6:
                        collision_risk_penalty -= (
                            closing_rate * proximity_factor**2 * 0.05
                        )

        # Total reward - removed steering_penalty and survival_bonus
        reward = (
            # speed_reward
            # + overtaking_reward
            distance_reward + proximity_penalty + speed_penalty + collision_risk_penalty
            # + crash_penalty
            # + completion_bonus
        )

        # Accumulate rewards throughout the episode
        self._accumulated_rewards["distance_reward"] = (
            distance_reward  # Current total distance reward
        )
        self._accumulated_rewards["crash_penalty"] += crash_penalty
        self._accumulated_rewards["completion_bonus"] += completion_bonus

        # Store both current step breakdown and accumulated totals
        self._reward_breakdown = {
            "distance_reward": distance_reward,
            "proximity_penalty": 0.0,  # Always 0 now
            "crash_penalty": crash_penalty,
            "completion_bonus": completion_bonus,
        }

        self._accumulated_breakdown = self._accumulated_rewards.copy()

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
            "accumulated_rewards": getattr(self, "_accumulated_breakdown", {}).copy(),
        }
        return info

    def close(self):
        pass

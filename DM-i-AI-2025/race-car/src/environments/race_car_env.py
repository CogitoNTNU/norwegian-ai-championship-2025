import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import src.game.core as core
from src.game.core import initialize_game_state, update_game, intersects


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
        self.games_per_batch = 1  # 3 games per batch
        self.current_step = 0
        self.current_game = 0
        self.crashed_steps = 0
        self.max_crashed_steps = 0  # End game instantly on crash

        # Batch tracking
        self.batch_rewards = []
        self.batch_distances = []

        # Pygame init (required for asset loading)
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

            if terminated:
                print(
                    f"Game {self.current_game + 1}/3 completed (CRASHED) - Distance: {core.STATE.distance:.1f}"
                )
            elif truncated:
                print(
                    f"Game {self.current_game + 1}/3 completed (TIMEOUT) - Distance: {core.STATE.distance:.1f}"
                )

            self.current_game += 1

            if self.current_game >= self.games_per_batch:
                # Batch complete - let PPO update
                total_distance = sum(self.batch_distances)
                avg_distance = total_distance / len(self.batch_distances)
                print(
                    f"BATCH COMPLETED! 3 games finished. Total distance: {total_distance:.1f}, Average: {avg_distance:.1f}"
                )
                print(
                    ">>> PPO UPDATE TRIGGERED - Model will now learn from the 3-game batch experience <<<"
                )
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
            normalized_distance = (
                min(sensor.reading, 1000.0) / 1000.0
                if sensor.reading is not None
                else 1.0
            )
            sensor_readings.append(normalized_distance)

        while len(sensor_readings) < 16:
            sensor_readings.append(1.0)
        sensor_readings = sensor_readings[:16]

        ego_car = core.STATE.ego
        velocity_x = np.clip(ego_car.velocity.x / 20.0, 0.0, 1.0)
        velocity_y = np.clip(abs(ego_car.velocity.y) / 10.0, 0.0, 1.0)
        position_y = ego_car.y / 1200.0
        lane_position = 0.5
        if ego_car.lane:
            lane_center = (ego_car.lane.y_start + ego_car.lane.y_end) / 2
            lane_position = np.clip((ego_car.y - lane_center + 120) / 240.0, 0.0, 1.0)

        observation = np.array(
            sensor_readings + [velocity_x, velocity_y, position_y, lane_position],
            dtype=np.float32,
        )
        return observation

    def _check_collisions(self):
        # Handle car collisions
        for car in core.STATE.cars:
            if car != core.STATE.ego and intersects(core.STATE.ego.rect, car.rect):
                core.STATE.crashed = True
                print("[CRASH] Collision with another car!")
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
                    print("[CRASH] Collision with wall!")
                    return

    def _calculate_reward(self, crashed_before):
        reward = 0.00  # Small base survival reward
        reward_breakdown = {"survival": 0.02}
        progress_reward = (
            core.STATE.distance - getattr(self, "_last_distance", 0)
        ) / 70.0
        self._last_distance = core.STATE.distance
        reward += progress_reward
        reward_breakdown["distance"] = progress_reward

        speed = core.STATE.ego.velocity.x
        speed_reward = 0
        if speed > 15:
            speed_reward = 0.15
        elif speed > 10:
            speed_reward = 0.1
        elif speed > 5:
            speed_reward = 0.05
        elif speed > 1:
            speed_reward = -0.01
        else:
            speed_reward = -0.05
        reward += speed_reward
        reward_breakdown["speed"] = speed_reward

        lane_reward = 0
        if core.STATE.ego.lane:
            lane_center = (core.STATE.ego.lane.y_start + core.STATE.ego.lane.y_end) / 2
            distance_from_center = abs(core.STATE.ego.y - lane_center)
            max_lane_deviation = 120
            if distance_from_center > max_lane_deviation:
                lane_reward = -0.05
        reward += lane_reward
        reward_breakdown["lane_position"] = lane_reward

        cars_ahead = 0
        closest_ahead_distance = float("inf")
        for car in core.STATE.cars:
            if car != core.STATE.ego:
                x_diff = car.x - core.STATE.ego.x
                y_diff = abs(car.y - core.STATE.ego.y)
                total_distance = abs(x_diff) + y_diff
                if x_diff > 0 and x_diff < 300:
                    cars_ahead += 1
                    closest_ahead_distance = min(closest_ahead_distance, total_distance)
        proximity_penalty = 0
        if cars_ahead > 0 and closest_ahead_distance < 200:
            if closest_ahead_distance < 100:
                proximity_penalty = -0.1
            elif closest_ahead_distance < 150:
                proximity_penalty = -0.05
            else:
                proximity_penalty = -0.02
        reward += proximity_penalty
        reward_breakdown["following_penalty"] = proximity_penalty

        # Collision avoidance, crash, and completion logic remain unchanged...
        # [Truncated for brevity, paste your own code for sensors/collision/bonus]

        # Crash penalty
        crash_penalty = 0
        if core.STATE.crashed and not crashed_before:
            crash_penalty = -1000.0
        elif core.STATE.crashed:
            crash_penalty = 0
        reward += crash_penalty
        reward_breakdown["crash_penalty"] = crash_penalty

        # Completion bonus
        completion_bonus = 0
        if self.current_step >= self.max_steps_per_game and not core.STATE.crashed:
            completion_bonus = 1000.0
        reward += completion_bonus
        reward_breakdown["completion_bonus"] = completion_bonus

        # Always add breakdown to info for logging
        self._reward_breakdown = reward_breakdown
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

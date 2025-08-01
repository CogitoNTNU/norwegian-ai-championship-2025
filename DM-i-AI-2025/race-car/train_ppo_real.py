import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
import torch
import wandb
import gymnasium as gym
from gymnasium import spaces
import pygame
import src.game.core as core
from src.game.core import initialize_game_state, update_game, intersects


class RealRaceCarEnv(gym.Env):
    """
    PPO environment that uses the actual race car game logic.
    """

    def __init__(self, seed_value=None, headless=True):
        super().__init__()

        self.seed_value = seed_value
        self.headless = headless
        self.max_steps = 3600  # 60 seconds at 60 FPS
        self.current_step = 0

        # Initialize pygame (required for asset loading)
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

        # Use the actual game initialization
        initialize_game_state("dummy_url", self.seed_value)
        self.current_step = 0
        self._last_distance = core.STATE.distance

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action_idx = int(action) if hasattr(action, "__iter__") else action
        action_str = self.action_map[action_idx]

        # Store crash state before update
        crashed_before = core.STATE.crashed

        # Use the actual game update logic (includes collision detection)
        update_game(action_str)

        # Check for collisions using the same logic as the actual game
        self._check_collisions()

        self.current_step += 1

        obs = self._get_observation()
        reward = self._calculate_reward(crashed_before)
        terminated = core.STATE.crashed
        truncated = (
            self.current_step >= self.max_steps
        )  # Only truncate after full 60 seconds
        info = self._get_info()

        # Log completion if full race finished without crash
        if truncated and not terminated:
            print(
                f"RACE COMPLETED! Full 60 seconds at step {self.current_step}, Distance: {core.STATE.distance:.1f}"
            )

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation using actual sensor data."""
        sensor_readings = []
        for sensor in core.STATE.sensors:
            if sensor.reading is not None:
                normalized_distance = min(sensor.reading, 1000.0) / 1000.0
            else:
                normalized_distance = 1.0
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
            lane_position = (ego_car.y - lane_center + 120) / 240.0
            lane_position = np.clip(lane_position, 0.0, 1.0)

        observation = np.array(
            sensor_readings + [velocity_x, velocity_y, position_y, lane_position],
            dtype=np.float32,
        )
        return observation

    def _check_collisions(self):
        """Check for collisions using the exact same logic as the actual game."""
        # Handle collisions with other cars
        for car in core.STATE.cars:
            if car != core.STATE.ego and intersects(core.STATE.ego.rect, car.rect):
                core.STATE.crashed = True
                return

        # Check collision with walls
        for wall in core.STATE.road.walls:
            if intersects(core.STATE.ego.rect, wall.rect):
                core.STATE.crashed = True
                return

    def _calculate_reward(self, crashed_before):
        """Calculate reward with proper crash detection."""
        reward = 0.01  # Base survival reward

        # Distance reward
        if hasattr(self, "_last_distance"):
            distance_reward = (core.STATE.distance - self._last_distance) / 100.0
            reward += distance_reward
        self._last_distance = core.STATE.distance

        # Speed reward
        speed = core.STATE.ego.velocity.x
        if 8 <= speed <= 15:
            reward += 0.1
        elif speed > 15:
            reward += 0.05
        elif speed < 5:
            reward -= 0.1

        # Lane centering
        if core.STATE.ego.lane:
            lane_center = (core.STATE.ego.lane.y_start + core.STATE.ego.lane.y_end) / 2
            distance_from_center = abs(core.STATE.ego.y - lane_center)
            max_lane_deviation = 60

            if distance_from_center < max_lane_deviation * 0.3:
                reward += 0.05
            elif distance_from_center < max_lane_deviation * 0.7:
                reward += 0.02
            else:
                reward -= 0.05

        # Collision avoidance
        min_distance_to_car = float("inf")
        for car in core.STATE.cars:
            if car != core.STATE.ego:
                distance = abs(car.x - core.STATE.ego.x) + abs(car.y - core.STATE.ego.y)
                min_distance_to_car = min(min_distance_to_car, distance)

        if min_distance_to_car < 100:
            reward -= 1.0
        elif min_distance_to_car < 200:
            reward -= 0.2

        # Crash penalty - only apply when crash just happened
        if core.STATE.crashed and not crashed_before:
            reward -= 50.0
            print(
                f"CRASH DETECTED at step {self.current_step}! Distance: {core.STATE.distance:.1f}"
            )

        # Huge bonus for completing full race without crashing
        if self.current_step >= self.max_steps and not core.STATE.crashed:
            reward += 1000.0  # Massive completion bonus
            print(
                f"RACE COMPLETION BONUS! Full 60 seconds, Distance: {core.STATE.distance:.1f}"
            )

        return reward

    def _get_info(self):
        return {
            "distance": core.STATE.distance,
            "speed": core.STATE.ego.velocity.x,
            "crashed": core.STATE.crashed,
            "ticks": core.STATE.ticks,
            "cars_nearby": len([c for c in core.STATE.cars if c != core.STATE.ego]),
            "race_completed": self.current_step >= self.max_steps
            and not core.STATE.crashed,
            "time_remaining": max(0, self.max_steps - self.current_step)
            / 60,  # seconds left
        }

    def close(self):
        pass


def train_real_ppo_model(
    project_name="race-car-real-ppo",
    run_name=None,
    timesteps=100000,
    use_wandb=True,
    resume_from=None,
):
    """Train PPO model using the real race car game."""

    n_envs = 4
    eval_freq = 10_000

    if use_wandb:
        config = {
            "algorithm": "PPO",
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "learning_rate": 3e-4,
            "reward_threshold": 200,
            "environment": "real_race_car_game",
        }

        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=["ppo", "race-car", "real-game", "rl"],
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Creating training environments using REAL race car game...")

    def make_env(rank=0):
        env = RealRaceCarEnv(seed_value=np.random.randint(0, 10000), headless=True)
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = Monitor(RealRaceCarEnv(seed_value=42, headless=True))

    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"Loading existing model from {resume_from}...")
        model = PPO.load(resume_from, env=train_env)
        print(f"Resumed from checkpoint: {resume_from}")
        print(f"Model will train for {timesteps:,} additional timesteps")
    else:
        print("Initializing new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print(f"Using device: {model.device}")

    # Callbacks
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=1200, verbose=1
    )  # ~Full race completion
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best,
        verbose=1,
    )

    print(f"Starting REAL game training for {timesteps:,} timesteps...")

    # Train
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    # Save final model
    final_model_path = "models/ppo_racecar_real_final"
    model.save(final_model_path)
    print("Real game training completed!")
    print(f"Final model saved to: {final_model_path}.zip")

    # Also save a checkpoint with timestamp for easy resuming
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"models/checkpoint_{timestamp}"
    model.save(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}.zip")

    # Clean up
    train_env.close()
    eval_env.close()

    if use_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train PPO model using REAL Race Car game"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Training timesteps"
    )
    parser.add_argument(
        "--project", default="race-car-real-ppo", help="Wandb project name"
    )
    parser.add_argument("--run-name", default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to model checkpoint to resume from (without .zip extension)",
    )

    args = parser.parse_args()

    print("Starting REAL PPO training for Race Car...")
    print("Using actual game elements, collision detection, and termination logic")

    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")

    model = train_real_ppo_model(
        project_name=args.project,
        run_name=args.run_name,
        timesteps=args.timesteps,
        use_wandb=not args.no_wandb,
        resume_from=args.resume_from,
    )

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback,
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
        self.crashed_steps = 0  # Steps since crash
        self.max_crashed_steps = 75  # Continue for 60 steps after crash (1 second)

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

        # Match real game experience: start with empty road, normal speed
        # Don't spawn traffic immediately - let it build up naturally
        core.STATE.cars = [core.STATE.ego]  # Only ego car initially

        self.current_step = 0
        self.crashed_steps = 0  # Reset crash counter
        self._last_distance = core.STATE.distance

        # Initialize tracking variables for aggressive driving rewards
        self._following_steps = 0
        self._last_y = core.STATE.ego.y

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action_idx = int(action) if hasattr(action, "__iter__") else action
        action_str = self.action_map[action_idx]

        # Store crash state before update
        crashed_before = core.STATE.crashed

        # If already crashed, don't update game state, just increment crash counter
        if core.STATE.crashed:
            self.crashed_steps += 1
        else:
            # Use the actual game update logic (includes collision detection)
            update_game(action_str)
            # Check for collisions using the same logic as the actual game
            self._check_collisions()

        self.current_step += 1

        obs = self._get_observation()
        reward = self._calculate_reward(crashed_before)

        # Only terminate after agent has experienced crash penalty for several steps
        terminated = core.STATE.crashed and self.crashed_steps >= self.max_crashed_steps
        truncated = (
            self.current_step >= self.max_steps
        )  # Only truncate after full 60 seconds
        info = self._get_info()

        # Log completion if full race finished without crash (keep this one)
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
        """Calculate reward encouraging aggressive driving and overtaking."""
        reward = 0.02  # Small base survival reward
        reward_breakdown = {"survival": 0.02}

        # Progress reward - main driving incentive
        progress_reward = 0
        if hasattr(self, "_last_distance"):
            progress_reward = (
                core.STATE.distance - self._last_distance
            ) / 100.0  # Smaller but steady
        self._last_distance = core.STATE.distance

        reward += progress_reward
        reward_breakdown["distance"] = progress_reward

        # Gradual speed reward - encourage forward movement
        speed = core.STATE.ego.velocity.x
        speed_reward = 0
        if speed > 15:
            speed_reward = 0.15  # Small bonus for high speed
        elif speed > 10:
            speed_reward = 0.1  # Small bonus for good speed
        elif speed > 5:
            speed_reward = 0.05  # Tiny bonus for movement
        elif speed > 1:
            speed_reward = -0.01  # Tiny penalty for slow movement
        else:
            speed_reward = -0.05  # Small penalty for stopping
        reward += speed_reward
        reward_breakdown["speed"] = speed_reward

        # Lane position - only penalize if completely off-road (wall collision territory)
        lane_reward = 0
        if core.STATE.ego.lane:
            lane_center = (core.STATE.ego.lane.y_start + core.STATE.ego.lane.y_end) / 2
            distance_from_center = abs(core.STATE.ego.y - lane_center)
            max_lane_deviation = 120  # Much more lenient

            # Only penalty for being dangerously close to walls
            if distance_from_center > max_lane_deviation:
                lane_reward = -0.05  # Penalty only for dangerous positioning
        reward += lane_reward
        reward_breakdown["lane_position"] = lane_reward

        # Smart collision avoidance and overtaking rewards
        cars_ahead = 0
        cars_behind = 0
        closest_ahead_distance = float("inf")
        closest_behind_distance = float("inf")

        for car in core.STATE.cars:
            if car != core.STATE.ego:
                x_diff = car.x - core.STATE.ego.x
                y_diff = abs(car.y - core.STATE.ego.y)
                total_distance = abs(x_diff) + y_diff

                # Cars ahead (in front)
                if x_diff > 0 and x_diff < 300:  # Within reasonable distance ahead
                    cars_ahead += 1
                    closest_ahead_distance = min(closest_ahead_distance, total_distance)

                # Cars behind
                elif x_diff < 0 and abs(x_diff) < 200:  # Cars we've passed recently
                    cars_behind += 1
                    closest_behind_distance = min(
                        closest_behind_distance, total_distance
                    )

        # Removed overtaking reward - focus on basic safe driving first
        overtaking_reward = (
            0  # No reward for overtaking until agent learns basic safety
        )
        reward += overtaking_reward
        reward_breakdown["overtaking"] = overtaking_reward

        # Car proximity penalty - discourage getting close to other cars
        proximity_penalty = 0
        if cars_ahead > 0 and closest_ahead_distance < 200:
            # Strong penalty for getting close to other cars
            if closest_ahead_distance < 100:
                proximity_penalty = -0.1  # Very close - dangerous
            elif closest_ahead_distance < 150:
                proximity_penalty = -0.05  # Close - concerning
            else:
                proximity_penalty = -0.02  # Getting close - caution
        reward += proximity_penalty
        reward_breakdown["following_penalty"] = (
            proximity_penalty  # Keep same key for logging
        )

        # Removed lane changing reward - can encourage dangerous maneuvers
        lane_change_reward = (
            0  # No reward for lane changes until basic safety is learned
        )
        if hasattr(self, "_last_y"):
            pass  # Still track position but don't reward changes
        else:
            self._last_y = core.STATE.ego.y
        self._last_y = core.STATE.ego.y
        reward += lane_change_reward
        reward_breakdown["lane_change"] = lane_change_reward

        # Smart collision avoidance using sensor data - only care about threats ahead/sides
        collision_avoidance_reward = 0

        # Get front and side sensor readings (the ones that matter for driving)
        front_sensors = []
        side_sensors = []

        for sensor in core.STATE.sensors:
            if sensor.reading is not None:
                # Front sensors (ahead of car) - these are the dangerous ones
                if sensor.name in [
                    "front",
                    "front_left_front",
                    "front_right_front",
                    "left_front",
                    "right_front",
                ]:
                    front_sensors.append(sensor.reading)
                # Side sensors (for lane changes) - moderate concern
                elif sensor.name in [
                    "left_side",
                    "right_side",
                    "left_side_front",
                    "right_side_front",
                ]:
                    side_sensors.append(sensor.reading)

        # Penalize based on closest front obstacle (immediate danger)
        if front_sensors:
            min_front_distance = min(front_sensors)
            if min_front_distance < 50:  # Very close ahead - dangerous
                collision_avoidance_reward -= 0.1
            elif min_front_distance < 100:  # Close ahead - concerning
                collision_avoidance_reward -= 0.05
            elif min_front_distance < 200:  # Moderately close
                collision_avoidance_reward -= 0.01
            elif min_front_distance > 300:  # Good forward clearance
                collision_avoidance_reward += 0.02  # Small reward for safe driving

        # Moderate penalty for side obstacles (affects lane changes)
        if side_sensors:
            min_side_distance = min(side_sensors)
            if min_side_distance < 60:  # Very close to sides
                collision_avoidance_reward -= 0.05
            elif min_side_distance < 120:  # Close to sides
                collision_avoidance_reward -= 0.02

        reward += collision_avoidance_reward
        reward_breakdown["collision_avoidance"] = collision_avoidance_reward

        # Crash penalty - apply continuously while crashed
        crash_penalty = 0
        if core.STATE.crashed and not crashed_before:
            # Initial crash penalty - noticeable but not overwhelming
            crash_penalty = -50.0
        elif core.STATE.crashed:
            # Continued penalty while staying crashed
            crash_penalty = 0  # Small continuous penalty
        reward += crash_penalty
        reward_breakdown["crash_penalty"] = crash_penalty

        # Huge bonus for completing full race without crashing
        completion_bonus = 0
        if self.current_step >= self.max_steps and not core.STATE.crashed:
            completion_bonus = 1000.0  # Massive completion bonus
        reward += completion_bonus
        reward_breakdown["completion_bonus"] = completion_bonus

        # Store reward breakdown for logging
        self._reward_breakdown = reward_breakdown

        return reward

    def _get_info(self):
        info = {
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

        # Add reward breakdown to info if available
        if hasattr(self, "_reward_breakdown"):
            info["reward_breakdown"] = self._reward_breakdown.copy()

        return info

    def close(self):
        pass


class WandbCallback(BaseCallback):
    """Custom callback for logging to wandb."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_lengths = []
        self.episode_crashes = []
        self.episode_completions = []
        self.reward_components = {
            "survival": [],
            "distance": [],
            "speed": [],
            "lane_position": [],
            "overtaking": [],
            "following_penalty": [],
            "lane_change": [],
            "collision_avoidance": [],
            "crash_penalty": [],
            "completion_bonus": [],
        }

    def _on_step(self) -> bool:
        # Collect episode data when episodes end
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    # Store episode metrics
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

                # Collect environment metrics from completed episodes
                if "distance" in info and (
                    info.get("crashed", False) or info.get("race_completed", False)
                ):
                    self.episode_distances.append(info["distance"])
                    self.episode_crashes.append(int(info["crashed"]))
                    self.episode_completions.append(int(info["race_completed"]))

                    # Collect crash data for wandb logging (but don't force rollout end)

        # Collect reward breakdown from environments - check all possible env structures
        if hasattr(self, "training_env"):
            # Handle vectorized environments
            if hasattr(self.training_env, "envs"):
                for env_idx, env in enumerate(self.training_env.envs):
                    # Check different nesting levels
                    env_obj = env
                    if hasattr(env, "env"):
                        env_obj = env.env
                    if hasattr(env_obj, "env"):
                        env_obj = env_obj.env

                    if hasattr(env_obj, "_reward_breakdown"):
                        for key, value in env_obj._reward_breakdown.items():
                            if key in self.reward_components:
                                self.reward_components[key].append(value)

        # Also try to collect from locals (direct environment access)
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "reward_breakdown" in info:
                    for key, value in info["reward_breakdown"].items():
                        if key in self.reward_components:
                            self.reward_components[key].append(value)

        # Removed forced rollout termination to allow full crash experience
        # This ensures the agent learns from the full consequences of crashes

        return True  # Always continue training

    def _on_rollout_end(self) -> None:
        # Log aggregated metrics at end of each rollout
        metrics = {}

        # Log both timesteps and training rounds for clarity
        training_round = (
            self.num_timesteps // 2048
        )  # Each round = 2048 timesteps (standard)
        metrics["Training/Round_Number"] = training_round
        metrics["Training/Total_Timesteps"] = self.num_timesteps

        # Log training losses and standard metrics
        if hasattr(self.model, "logger") and self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith(("train/", "rollout/")):
                    metrics[key] = value

        # Log episode statistics with clear names
        if self.episode_rewards:
            metrics["Performance/Average_Episode_Reward"] = np.mean(
                self.episode_rewards
            )
            metrics["Performance/Reward_Consistency"] = (
                np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0
            )
            metrics["Performance/Episodes_This_Round"] = len(self.episode_rewards)
            print(
                f"Logging {len(self.episode_rewards)} episode rewards, mean: {np.mean(self.episode_rewards):.2f}"
            )
            self.episode_rewards.clear()

        if self.episode_lengths:
            metrics["Performance/Average_Episode_Length"] = np.mean(
                self.episode_lengths
            )
            self.episode_lengths.clear()

        if self.episode_distances:
            metrics["Racing/Average_Distance_Traveled"] = np.mean(
                self.episode_distances
            )
            metrics["Racing/Best_Distance_This_Round"] = np.max(self.episode_distances)
            self.episode_distances.clear()

        if self.episode_crashes:
            metrics["Racing/Crash_Rate_Percent"] = np.mean(self.episode_crashes) * 100
            self.episode_crashes.clear()

        if self.episode_completions:
            metrics["Racing/Race_Completion_Rate_Percent"] = (
                np.mean(self.episode_completions) * 100
            )
            self.episode_completions.clear()

        # Log reward breakdown with clear names
        reward_mapping = {
            "survival": "Reward_Components/Survival_Bonus",
            "distance": "Reward_Components/Distance_Progress",
            "speed": "Reward_Components/Speed_Bonus",
            "lane_position": "Reward_Components/Lane_Positioning",
            "overtaking": "Reward_Components/Overtaking_Bonus",
            "following_penalty": "Reward_Components/Following_Penalty",
            "lane_change": "Reward_Components/Lane_Change_Bonus",
            "collision_avoidance": "Reward_Components/Collision_Avoidance",
            "crash_penalty": "Reward_Components/Crash_Penalty",
            "completion_bonus": "Reward_Components/Completion_Bonus",
        }

        reward_components_logged = 0
        for component, values in self.reward_components.items():
            if values and component in reward_mapping:
                metrics[reward_mapping[component]] = np.mean(values)
                reward_components_logged += 1
                values.clear()

        print(
            f"Logged {reward_components_logged} reward components out of {len(self.reward_components)}"
        )
        if reward_components_logged == 0:
            print("WARNING: No reward components were logged!")
            # Log some debug info
            for component, values in self.reward_components.items():
                print(f"  {component}: {len(values)} values")

        # Always log even if some data is missing
        print(f"Round {training_round}: Logging {len(metrics)} metrics to wandb")

        # Use training round as the step for cleaner x-axis in wandb
        wandb.log(metrics, step=training_round)


def train_real_ppo_model(
    project_name="race-car-real-ppo",
    run_name=None,
    timesteps=100000,
    training_rounds=None,
    use_wandb=True,
    resume_from=None,
):
    """Train PPO model using the real race car game."""

    n_envs = 4
    eval_freq = 10_000

    # Convert training rounds to timesteps if specified
    if training_rounds is not None:
        n_steps = 75  # Standard PPO n_steps - crashes force immediate updates
        timesteps = training_rounds * n_steps
        print(f"Training for {training_rounds} rounds ({timesteps:,} timesteps)")
        print("Crashes trigger immediate updates regardless of round size")
    else:
        print(f"Training for {timesteps:,} timesteps")

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
            n_steps=75,
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

    callbacks = [eval_callback]

    if use_wandb:
        wandb_callback = WandbCallback(verbose=1)
        callbacks.append(wandb_callback)

    print(f"Starting REAL game training for {timesteps:,} timesteps...")

    # Train
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

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
        "--rounds",
        type=int,
        default=None,
        help="Training rounds (each round = 2048 timesteps)",
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
        training_rounds=args.rounds,
        use_wandb=not args.no_wandb,
        resume_from=args.resume_from,
    )

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
        for car in core.STATE.cars:
            if car != core.STATE.ego and intersects(core.STATE.ego.rect, car.rect):
                core.STATE.crashed = True
                print("[CRASH] Collision with another car!")
                return
        for wall in core.STATE.road.walls:
            if intersects(core.STATE.ego.rect, wall.rect):
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
        print("\n=== PPO OPTIMIZATION STARTING ===")
        print(
            f"Collected {self.model.n_steps} steps from {self.model.n_envs} environments"
        )
        print(f"Total timesteps so far: {self.num_timesteps}")
        print("Performing PPO update with collected batch data...")
        print("=" * 35 + "\n")

        metrics = {}

        # Log both timesteps and training rounds for clarity
        training_round = (
            self.num_timesteps // 30000
        )  # Each round = 2048 timesteps (standard)
        metrics["Training/Round_Number"] = training_round
        metrics["Training/Total_Timesteps"] = self.num_timesteps

        # Log training losses and standard metrics
        if hasattr(self.model, "logger") and self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith(("train/", "rollout/")):
                    metrics[key] = value

        # Log batch-specific metrics
        metrics["Training/Batch_Episodes_Completed"] = len(self.episode_rewards)

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

        print("=== PPO OPTIMIZATION COMPLETE ===")
        print(f"Update round {training_round} finished")
        print("Model weights updated based on batch experience")
        print("Resuming environment collection...\n")


def train_real_ppo_model(
    project_name="race-car-real-ppo-batch",
    run_name=None,
    timesteps=90000,  # Increased default to account for 3x longer episodes
    training_rounds=None,
    use_wandb=True,
    resume_from=None,
):
    """Train PPO model using the real race car game with 3-game batches."""

    n_envs = 4
    eval_freq = 50_000  # Adjusted for longer episodes

    # Convert training rounds to timesteps if specified
    if training_rounds is not None:
        # Each batch now contains 3 games, so episodes are ~3x longer
        n_steps = 3600  # 3 games * 3600 steps per game = 10800 steps per batch
        timesteps = training_rounds * n_steps
        print(f"Training for {training_rounds} rounds ({timesteps:,} timesteps)")
        print("Each batch contains 3 games of 1 minute each")
        print("PPO updates after every 3-game batch completion")
    else:
        print(f"Training for {timesteps:,} timesteps")
        print("Each episode is now a batch of 3 games (3 minutes total)")

    if use_wandb:
        config = {
            "algorithm": "PPO",
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "learning_rate": 3e-4,
            "reward_threshold": 600,  # Increased for 3-game batches
            "environment": "real_race_car_batch_game",
            "games_per_batch": 3,
            "game_duration_seconds": 60,
        }

        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=["ppo", "race-car", "real-game", "rl", "batch-training"],
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
            n_steps=2048,  # Collect more steps since episodes are longer (3 games)
            batch_size=64,
            n_epochs=10,
            gamma=0.99,  # Keep standard discount factor
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),  # Larger network for complex batch learning
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

    print(f"Starting REAL batch training for {timesteps:,} timesteps...")
    print("Each episode contains 3 consecutive 1-minute games")
    print("PPO updates after completing each 3-game batch")

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

    print("Starting REAL PPO training for Race Car with 3-game batches...")
    print(
        "Using actual game elements, collision detection, and batch termination logic"
    )
    print("Each training episode = 3 consecutive games of 1 minute each")

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

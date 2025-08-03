import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


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

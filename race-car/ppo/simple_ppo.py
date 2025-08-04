import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import wandb
from wandb.integration.sb3 import WandbCallback
from simple_race_env import RaceCarEnv
import numpy as np

class DistanceLoggingCallback(BaseCallback):
    """Custom callback for logging distance traveled to wandb."""
    
    def __init__(self, verbose=0):
        super(DistanceLoggingCallback, self).__init__(verbose)
        self.episode_distances = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Check if any episode finished
        if len(self.locals.get("dones", [])) > 0:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    # Get the info from the environment
                    info = self.locals["infos"][i]
                    if "episode" in info:
                        # Log episode statistics
                        distance = info["episode"].get("distance", 0)
                        reward = info["episode"].get("r", 0)
                        length = info["episode"].get("l", 0)
                        
                        self.episode_distances.append(distance)
                        self.episode_rewards.append(reward)
                        
                        # Log to wandb
                        wandb.log({
                            "episode/distance": distance,
                            "episode/reward": reward,
                            "episode/length": length,
                            "episode/avg_distance_last_100": np.mean(self.episode_distances[-100:]) if self.episode_distances else 0,
                            "episode/avg_reward_last_100": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                            "global_step": self.num_timesteps
                        })
                        
                        if self.verbose > 0:
                            print(f"Episode finished - Distance: {distance:.1f}, Reward: {reward:.1f}, Length: {length}")
        
        return True

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100_000,
    "env_name": "RaceCar",
}

run = wandb.init(
    project="test",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    entity="NM-i-KI"
)

def make_env():
    env = RaceCarEnv(render_mode="rgb_array")
    env = Monitor(env)  # record stats such as returns
    # Wrap with RecordVideo
    env = RecordVideo(
        env,
        f"videos/{run.id}",
        episode_trigger=lambda episode_id: episode_id % 10 == 0,  # Record every 10th episode
        disable_logger=True  # Optional: disable the default logger
    )
    return env

env = make_env()

# Create the model
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

# Combine callbacks
callbacks = [
    WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
    DistanceLoggingCallback(verbose=1)
]

# Train the model
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callbacks
)

run.finish()

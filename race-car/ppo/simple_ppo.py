import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import wandb
from wandb.integration.sb3 import WandbCallback
from simple_race_env import RaceCarEnv

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

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

run.finish()

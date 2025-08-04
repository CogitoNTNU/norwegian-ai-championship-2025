import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from race_car_gym_env import RaceCarEnv


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100_000,
    "env_name": "Rac",
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
    return env


env = SubprocVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)
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

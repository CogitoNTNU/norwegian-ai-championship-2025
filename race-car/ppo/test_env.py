from typing import Optional
from race_car_gym_env import RaceCarEnv


def test_race_car_env(render_mode: Optional[str]):
# Test the environment
    print("Testing Race Car Gym Environment...")

# Create environment without rendering
    env = RaceCarEnv(render_mode=render_mode)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Number of sensors: {env.num_sensors}")

# Reset environment
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

# Run a few random steps
    total_reward = 0
    for i in range(10_000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            print(f"\nStep {i}:")
            print(f"  Action: {env.action_map[action]}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Distance: {info['distance']:.1f}")
            print(f"  Velocity X: {info['velocity_x']:.1f}")
            print(f"  Reward breakdown: {info['reward_breakdown']}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {i}")
            print(f"  Final distance: {info['distance']:.1f}")
            print(f"  Total reward: {total_reward:.1f}")
            print(f"  Crashed: {info['crashed']}")
            break

    env.close()
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    test_race_car_env(None)
    test_race_car_env("human")

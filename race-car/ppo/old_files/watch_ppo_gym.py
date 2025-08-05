import os
from stable_baselines3 import PPO
import pygame
from race_car_gym_env import RaceCarEnv
import time


def watch_ppo_model(model_path, num_episodes=5, seed=None):
    """Watch a trained PPO model play the race car game."""

    # Check if model exists
    if not os.path.exists(model_path):
        if not model_path.endswith(".zip"):
            model_path = model_path + ".zip"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return

    print(f"Loading model from: {model_path}")

    # Create environment with rendering
    env = RaceCarEnv(render_mode="human", seed=seed or "watch_seed")

    # Load the model
    model = PPO.load(model_path, env=env)

    print("Model loaded successfully!")
    print(f"Action space: {env.action_space}")
    print("Actions: 0=NOTHING, 1=ACCELERATE, 2=DECELERATE")
    print(f"\nWatching {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step_count += 1

            # Small delay to make viewing easier
            time.sleep(0.016)  # ~60 FPS

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
                    elif event.key == pygame.K_SPACE:
                        # Pause on spacebar
                        paused = True
                        while paused:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.KEYDOWN:
                                    if pause_event.key == pygame.K_SPACE:
                                        paused = False
                                elif pause_event.type == pygame.QUIT:
                                    env.close()
                                    return
                            time.sleep(0.1)

        print("Episode finished:")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Distance: {info.get('distance', 0):.2f}")
        print(f"  Crashed: {info.get('crashed', False)}")
        print(f"  Final velocity: {info.get('velocity_x', 0):.2f}")

        # Wait a bit between episodes
        time.sleep(1)

    env.close()
    print("\nDone watching!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch trained PPO model play Race Car (No Steering)"
    )
    parser.add_argument(
        "model_path", help="Path to the trained model (with or without .zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to watch"
    )
    parser.add_argument(
        "--seed", type=str, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=== PPO Race Car Viewer (No Steering) ===")
    print("Controls:")
    print("  SPACE: Pause/unpause")
    print("  ESC: Exit")
    print()

    watch_ppo_model(args.model_path, args.episodes, args.seed)

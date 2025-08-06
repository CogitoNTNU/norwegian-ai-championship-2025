#!/usr/bin/env python3
"""
Watch bot_with_memory controller in the PPO environment.
This script uses the existing bot_with_memory.py LaneChangeController
to control the race car in the PPO gym environment.
"""

import sys
import os
import pygame
import numpy as np
import time
import argparse

# Add ppo directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "ppo"))

from src.environments.race_car_gym_env import RaceCarEnv
from rule_based_AI.bot_with_memory import LaneChangeController


def get_sensor_data_from_observation(observation: np.ndarray) -> dict:
    """Convert gym observation to bot_with_memory format."""
    num_sensors = len(observation) - 4

    # Convert normalized sensors back to distance readings
    sensors = {}

    # Sensor configuration based on typical setup
    sensor_names = [
        "front",
        "right_front",
        "right_side",
        "right_back",
        "back",
        "left_back",
        "left_side",
        "left_front",
        "left_side_front",
        "front_left_front",
        "front_right_front",
        "right_side_front",
        "right_side_back",
        "back_right_back",
        "back_left_back",
        "left_side_back",
    ]

    # Map sensor indices to names
    for i in range(min(num_sensors, len(sensor_names))):
        # Convert from normalized (-1 to 1) to distance (0 to 1000)
        distance = float((observation[i] + 1.0) * 500.0)  # Convert to Python float
        sensors[sensor_names[i]] = distance

        # Also add numbered versions for compatibility
        sensors[f"sensor_{i}"] = distance
        sensors[str(i)] = distance

    # Extract velocity
    vel_x = float(
        observation[num_sensors] * 30.0
    )  # Denormalize and convert to Python float
    vel_y = float(observation[num_sensors + 1] * 10.0)  # Convert to Python float

    return {
        "sensors": sensors,
        "velocity": {"x": vel_x, "y": vel_y},
        "did_crash": False,
    }


def watch_bot_in_ppo_env(
    episodes: int = 1, render_speed: float = 1.0, verbose: bool = True
):
    """
    Watch the bot_with_memory controller in the PPO environment.
    """
    print("🏁 Starting Bot with Memory Demo in PPO Environment")
    print("=" * 50)

    # Stats tracking
    episode_stats = []

    # Create environment
    try:
        env = RaceCarEnv(render_mode="human", seed=None)
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    # Create controller instance
    controller = LaneChangeController(verbose=verbose)

    try:
        for episode in range(episodes):
            print(f"\n🏎️  Episode {episode + 1}/{episodes}")
            print("-" * 30)

            # Reset environment
            try:
                obs, info = env.reset()
            except Exception as e:
                print(f"❌ Error resetting environment: {e}")
                continue

            done = False
            episode_distance = 0
            episode_reward = 0
            episode_steps = 0
            action_counts = {
                "NOTHING": 0,
                "ACCELERATE": 0,
                "DECELERATE": 0,
                "STEER_LEFT": 0,
                "STEER_RIGHT": 0,
            }

            # Action mapping from string to gym action
            action_to_gym = {
                "NOTHING": 0,
                "ACCELERATE": 1,
                "DECELERATE": 2,
                "STEER_LEFT": 3,
                "STEER_RIGHT": 4,
            }

            start_time = time.time()
            current_action_queue = []

            while not done:
                try:
                    # If we have queued actions, use them
                    if current_action_queue:
                        action_str = current_action_queue.pop(0)
                    else:
                        # Get sensor data in bot format
                        request_data = get_sensor_data_from_observation(obs)
                        request_data["did_crash"] = info.get("crashed", False)

                        # Get actions from controller
                        actions = controller.predict_actions(request_data)
                        current_action_queue = actions.copy()
                        action_str = (
                            current_action_queue.pop(0)
                            if current_action_queue
                            else "NOTHING"
                        )

                    # Convert to gym action
                    gym_action = action_to_gym.get(action_str, 0)
                    action_counts[action_str] = action_counts.get(action_str, 0) + 1

                    # Take action
                    obs, reward, terminated, truncated, info = env.step(gym_action)
                    done = terminated or truncated

                    episode_reward += reward
                    episode_steps += 1

                    # Control rendering speed
                    if render_speed < 1.0:
                        time.sleep((1.0 - render_speed) * 0.016)

                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            episodes = episode + 1
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                print("\n⏸️  PAUSED - Press SPACE to continue")
                                paused = True
                                while paused:
                                    for pause_event in pygame.event.get():
                                        if (
                                            pause_event.type == pygame.KEYDOWN
                                            and pause_event.key == pygame.K_SPACE
                                        ):
                                            paused = False
                                        elif pause_event.type == pygame.QUIT:
                                            paused = False
                                            done = True
                                    time.sleep(0.1)
                            elif event.key == pygame.K_r:
                                print("\n🔄 Resetting episode...")
                                done = True
                            elif event.key == pygame.K_ESCAPE:
                                print("\n🛑 Exiting...")
                                episodes = episode + 1
                                done = True

                except Exception as e:
                    print(f"❌ Error during episode: {e}")
                    done = True
                    break

            # Episode complete
            episode_time = time.time() - start_time
            episode_distance = info["distance"]

            print(f"\n✅ Episode {episode + 1} Complete!")
            print(f"  Final Distance: {episode_distance:.1f}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {episode_steps}")
            print(f"  Time: {episode_time:.1f}s")
            print(f"  Crashed: {'Yes' if info['crashed'] else 'No'}")

            # Action distribution
            print("\n📊 Action Distribution:")
            total_actions = sum(action_counts.values())
            for action_name, count in sorted(
                action_counts.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                if count > 0:
                    print(f"  {action_name}: {count} ({percentage:.1f}%)")

            episode_stats.append(
                {
                    "episode": episode + 1,
                    "distance": episode_distance,
                    "reward": episode_reward,
                    "steps": episode_steps,
                    "crashed": info["crashed"],
                    "time": episode_time,
                }
            )

    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")

    finally:
        # Summary statistics
        if episode_stats:
            print("\n" + "=" * 50)
            print("📈 Summary Statistics")
            print("=" * 50)

            distances = [s["distance"] for s in episode_stats]
            rewards = [s["reward"] for s in episode_stats]
            steps = [s["steps"] for s in episode_stats]
            crashes = [s["crashed"] for s in episode_stats]

            print(f"Episodes Run: {len(episode_stats)}")
            print(
                f"Average Distance: {np.mean(distances):.1f} ± {np.std(distances):.1f}"
            )
            print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"Average Steps: {np.mean(steps):.0f}")
            print(f"Crash Rate: {sum(crashes) / len(crashes) * 100:.1f}%")

            # Best episode
            best_episode = max(episode_stats, key=lambda x: x["distance"])
            print(f"\n🏆 Best Episode: #{best_episode['episode']}")
            print(f"  Distance: {best_episode['distance']:.1f}")

        controller.close()
        env.close()
        print("\n👋 Demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch bot_with_memory in PPO environment"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to watch"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Rendering speed multiplier"
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")

    args = parser.parse_args()

    print("🎮 Bot with Memory Demo in PPO Environment")
    print("\nThis demo shows the lane change controller in the PPO environment.")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  R - Reset current episode")
    print("  ESC - Exit")
    print("  Close Window - Quit")
    print("\nPress CTRL+C to stop early\n")

    watch_bot_in_ppo_env(
        episodes=args.episodes, render_speed=args.speed, verbose=not args.quiet
    )

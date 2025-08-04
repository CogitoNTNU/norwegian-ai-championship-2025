"""
Debug script to test if the environment is working correctly.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from race_car_gym_env import RaceCarEnv
import src.game.core as game_core


def test_environment():
    """Test the environment to ensure distance tracking works."""

    print("Testing Race Car Environment...")
    print("=" * 60)

    # Create environment
    env = RaceCarEnv(render_mode=None, seed="test_seed")

    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    print(f"Initial distance: {game_core.STATE.distance}")
    print(f"Initial velocity: {game_core.STATE.ego.velocity.x}")

    # Test different actions
    actions_to_test = [
        (1, "ACCELERATE"),
        (1, "ACCELERATE"),
        (1, "ACCELERATE"),
        (0, "NOTHING"),
        (0, "NOTHING"),
    ]

    print("\nTesting actions...")
    print("-" * 60)

    total_reward = 0
    for i, (action, action_name) in enumerate(actions_to_test):
        # Store state before action
        prev_distance = game_core.STATE.distance
        prev_velocity = game_core.STATE.ego.velocity.x

        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print results
        print(f"\nStep {i + 1}: {action_name}")
        print(
            f"  Velocity: {prev_velocity:.2f} -> {game_core.STATE.ego.velocity.x:.2f}"
        )
        print(f"  Distance: {prev_distance:.2f} -> {game_core.STATE.distance:.2f}")
        print(f"  Distance gained: {game_core.STATE.distance - prev_distance:.2f}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Info distance: {info.get('distance', 'NOT FOUND')}")
        print(f"  Crashed: {info.get('crashed', False)}")

        if terminated or truncated:
            print(f"\nEpisode ended early at step {i + 1}")
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Final distance: {game_core.STATE.distance:.2f}")

    # Test multiple episodes
    print("\n" + "=" * 60)
    print("Testing multiple episodes...")

    episode_distances = []
    episode_rewards = []

    for ep in range(5):
        obs, info = env.reset()  # This should reset distance to 0
        episode_reward = 0
        steps = 0

        # Run episode with random actions
        while steps < 100:  # Limit steps for testing
            action = 1 if steps < 50 else 0  # Accelerate then coast
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        final_distance = info.get("distance", 0)
        episode_distances.append(final_distance)
        episode_rewards.append(episode_reward)

        print(
            f"Episode {ep + 1}: Distance={final_distance:.2f}, Reward={episode_reward:.2f}, Steps={steps}"
        )

    print(f"\nAverage distance: {np.mean(episode_distances):.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")

    # Check if distances are reasonable (should be different from each other due to randomness)
    if all(d == episode_distances[0] for d in episode_distances):
        print("\nWARNING: All episodes have identical distance - check randomization!")

    # Check if distance is being tracked at all
    if all(d == 0 for d in episode_distances):
        print("\n" + "!" * 60)
        print("WARNING: Distance is always 0!")
        print("Possible issues:")
        print("1. The distance tracking in game_core might not be working")
        print("2. The environment might not be properly updating STATE.distance")
        print("3. The ego car velocity might be 0")
        print("!" * 60)

    env.close()


def test_game_core_directly():
    """Test the game core directly to see if distance tracking works."""

    print("\n" + "=" * 60)
    print("Testing game core directly...")

    from src.game.core import initialize_game_state, update_game, STATE

    # Initialize game
    initialize_game_state("", "test_seed")

    print(f"Initial STATE.distance: {STATE.distance}")
    print(f"Initial ego velocity: {STATE.ego.velocity.x}")
    print(f"Initial ego position: ({STATE.ego.x}, {STATE.ego.y})")

    # Update game with accelerate action
    for i in range(10):
        prev_distance = STATE.distance
        update_game("ACCELERATE")
        print(
            f"Step {i + 1}: Distance {prev_distance:.2f} -> {STATE.distance:.2f}, Velocity: {STATE.ego.velocity.x:.2f}"
        )

    if STATE.distance == 0:
        print("\nERROR: Distance is not being updated in game core!")
        print("The issue is in the core game logic, not the gym wrapper.")


if __name__ == "__main__":
    # First test the gym environment
    test_environment()

    # Then test the game core directly
    test_game_core_directly()

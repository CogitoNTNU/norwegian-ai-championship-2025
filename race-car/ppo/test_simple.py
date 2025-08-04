"""
Verify that the reset fix is working correctly.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from race_car_gym_env import RaceCarEnv
import src.game.core as game_core


def test_reset_fix():
    """Test that reset properly clears distance."""

    print("Testing reset fix...")
    print("=" * 60)

    env = RaceCarEnv(render_mode=None, seed="test")

    # First episode
    print("\nEpisode 1:")
    obs, info = env.reset()
    print(
        f"After reset: distance={game_core.STATE.distance}, info['distance']={info['distance']}"
    )

    # Take some steps
    for i in range(10):
        obs, reward, term, trunc, info = env.step(1)  # ACCELERATE

    print(
        f"After 10 steps: distance={game_core.STATE.distance:.1f}, info['distance']={info['distance']:.1f}"
    )

    # Second episode
    print("\nEpisode 2:")
    obs, info = env.reset()
    print(
        f"After reset: distance={game_core.STATE.distance}, info['distance']={info['distance']}"
    )

    if game_core.STATE.distance != 0:
        print("ERROR: Distance not reset to 0!")
        print("Make sure your reset method includes:")
        print("  game_core.STATE.distance = 0")
        print("  game_core.STATE.ticks = 0")
        print("  game_core.STATE.crashed = False")
    else:
        print("SUCCESS: Distance properly reset to 0!")

    # Take some steps in second episode
    for i in range(10):
        obs, reward, term, trunc, info = env.step(1)  # ACCELERATE

    print(
        f"After 10 steps: distance={game_core.STATE.distance:.1f}, info['distance']={info['distance']:.1f}"
    )

    # Third episode
    print("\nEpisode 3:")
    obs, info = env.reset()
    print(
        f"After reset: distance={game_core.STATE.distance}, info['distance']={info['distance']}"
    )

    env.close()

    print("\n" + "=" * 60)
    print("Summary:")
    print("If distance is 0 after each reset, the fix is working.")
    print("If distance persists between episodes, the reset method needs updating.")


if __name__ == "__main__":
    test_reset_fix()

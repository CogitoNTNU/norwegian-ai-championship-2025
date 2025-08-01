import random
import os


def return_action(state):
    """
    AI model implementation for race car actions.
    First tries to use trained PPO model, falls back to random if not available.
    """

    # Try to use PPO model if available
    if os.path.exists("models/best_model.zip"):
        try:
            from ppo_agent import return_action as ppo_return_action

            return ppo_return_action(state)
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            print("Falling back to random actions")

    # Fallback to random actions
    action_choices = [
        "ACCELERATE",
        "DECELERATE",
        "STEER_LEFT",
        "STEER_RIGHT",
        "NOTHING",
    ]

    return {
        "action_type": "QUEUE",
        "actions": [random.choice(action_choices) for _ in range(10)],
    }

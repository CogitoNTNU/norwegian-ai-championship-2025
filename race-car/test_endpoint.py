import random


def return_action(state):
    """
    Basic test implementation that returns random actions.
    Replace this with your AI model logic.
    """
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

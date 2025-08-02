import os
import numpy as np
from stable_baselines3 import PPO


def return_action(state):
    """
    PPO model inference for race car actions.
    Loads the best model and returns the predicted action.
    """
    try:
        # Load the trained model
        model_path = "../models/best_model.zip"
        if not os.path.exists(model_path):
            model_path = "models/best_model.zip"  # Fallback path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = PPO.load(model_path)

        # Convert state to numpy array if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        # Get action from model
        action, _ = model.predict(state, deterministic=True)

        # Convert action to string format expected by game
        action_map = {
            0: "NOTHING",
            1: "ACCELERATE",
            2: "DECELERATE",
            3: "STEER_LEFT",
            4: "STEER_RIGHT",
        }

        return action_map.get(int(action), "NOTHING")

    except Exception as e:
        print(f"Error in PPO agent: {e}")
        raise e

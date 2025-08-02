"""
PPO Agent for the race car game endpoint.
"""

import os
import numpy as np
from stable_baselines3 import PPO


# Global model instance to avoid reloading
_model = None
_model_path = None


def load_model(model_path="models/best_model.zip"):
    """Load the PPO model."""
    global _model, _model_path

    if _model is None or _model_path != model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading PPO model from {model_path}")
        _model = PPO.load(model_path.replace(".zip", ""))
        _model_path = model_path
        print("PPO model loaded successfully")

    return _model


def state_to_observation(state):
    """Convert game state to PPO observation format."""
    # Extract sensor readings
    sensor_readings = []
    sensors = state.get("sensors", [])

    for sensor in sensors:
        reading = sensor.get("reading")
        if reading is not None:
            normalized_distance = min(reading, 1000.0) / 1000.0
        else:
            normalized_distance = 1.0
        sensor_readings.append(normalized_distance)

    # Ensure we have 16 sensor values
    while len(sensor_readings) < 16:
        sensor_readings.append(1.0)
    sensor_readings = sensor_readings[:16]

    # Extract ego car information
    ego = state.get("ego", {})
    velocity = ego.get("velocity", {"x": 0, "y": 0})

    # Normalize velocity components
    velocity_x = np.clip(velocity["x"] / 20.0, 0.0, 1.0)
    velocity_y = np.clip(abs(velocity["y"]) / 10.0, 0.0, 1.0)

    # Normalize position
    position_y = ego.get("y", 600) / 1200.0

    # Lane position (simplified)
    lane_position = 0.5  # Default to center

    # Combine into observation
    observation = np.array(
        sensor_readings + [velocity_x, velocity_y, position_y, lane_position],
        dtype=np.float32,
    )

    return observation


def return_action(state):
    """
    Return action for the race car using PPO model.

    Args:
        state: Game state dictionary

    Returns:
        Action dictionary with action_type and actions list
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

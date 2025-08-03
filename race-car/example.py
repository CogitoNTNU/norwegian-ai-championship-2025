import pygame
import numpy as np
import os
from stable_baselines3 import PPO
from src.game.core import initialize_game_state, game_loop


"""
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
"""

# Load the trained PPO model
MODEL_PATH = "models/ppo_racecar_real_final.zip"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = PPO.load(MODEL_PATH)
        print(f"Loaded PPO model from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
else:
    print(f"Model file {MODEL_PATH} not found, falling back to random actions")


def predict_race_car_action(state):
    """
    Predict car actions using the trained PPO model.
    
    Args:
        state (dict): State dictionary containing:
            - did_crash (bool)
            - elapsed_time_ms (int)
            - distance (int)
            - velocity (dict): {"x": int, "y": int}
            - coordinates (dict): {"x": int, "y": int}
            - sensors (dict): sensor readings
    
    Returns:
        list: List of action strings
    """
    global model
    
    if model is None:
        # Fallback to random actions if model not loaded
        action_choices = [
            "ACCELERATE",
            "DECELERATE", 
            "STEER_LEFT",
            "STEER_RIGHT",
            "NOTHING",
        ]
        actions = []
        for _ in range(10):
            actions.append(np.random.choice(action_choices))
        return actions
    
    try:
        # Convert state to observation format expected by the model
        # The model expects a flattened observation array
        obs = []
        
        # Add crash status (0 or 1)
        obs.append(1.0 if state.get('did_crash', False) else 0.0)
        
        # Add normalized elapsed time (divide by 60000 for 60 seconds max)
        obs.append(state.get('elapsed_time_ms', 0) / 60000.0)
        
        # Add normalized distance
        obs.append(state.get('distance', 0) / 1000.0)  # Normalize by reasonable max distance
        
        # Add velocity components
        velocity = state.get('velocity', {'x': 0, 'y': 0})
        obs.append(velocity.get('x', 0) / 100.0)  # Normalize velocity
        obs.append(velocity.get('y', 0) / 100.0)
        
        # Add coordinates
        coordinates = state.get('coordinates', {'x': 0, 'y': 0})
        obs.append(coordinates.get('x', 0) / 1600.0)  # Normalize by screen width
        obs.append(coordinates.get('y', 0) / 1200.0)  # Normalize by screen height
        
        # Add sensor readings
        sensors = state.get('sensors', {})
        # Assuming we have 8 sensors (common for race car environments)
        for i in range(8):
            sensor_key = f'sensor_{i}'
            sensor_value = sensors.get(sensor_key, sensors.get(str(i), None))
            if sensor_value is None:
                obs.append(0.0)  # No detection
            else:
                obs.append(min(sensor_value / 100.0, 1.0))  # Normalize and cap at 1.0
        
        # Convert to numpy array
        observation = np.array(obs, dtype=np.float32)
        
        # Get action from model
        action, _states = model.predict(observation, deterministic=True)
        
        # Convert action index to action string
        action_map = {
            0: "ACCELERATE",
            1: "DECELERATE", 
            2: "STEER_LEFT",
            3: "STEER_RIGHT",
            4: "NOTHING",
        }
        
        predicted_action = action_map.get(action, "NOTHING")
        
        # Return list of actions (for consistency with API)
        return [predicted_action] * 10
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Fallback to random actions on error
        action_choices = [
            "ACCELERATE",
            "DECELERATE",
            "STEER_LEFT", 
            "STEER_RIGHT",
            "NOTHING",
        ]
        actions = []
        for _ in range(10):
            actions.append(np.random.choice(action_choices))
        return actions


if __name__ == "__main__":
    seed_value = 12345
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(verbose=True)  # For pygame window
    pygame.quit()

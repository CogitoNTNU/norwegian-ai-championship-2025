"""
Benchmark bot_with_memory controller in the PPO environment.
This script runs the bot across 20 predefined seeds for consistent testing.
"""

import sys
import os
import numpy as np
import time
import json
from datetime import datetime
from statistics import median  # Added for median calculations

# Add ppo directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "ppo"))

from ppo.race_car_gym_env import RaceCarEnv
from bot_with_memory import LaneChangeController


# Predefined seeds for consistent benchmarking
BENCHMARK_SEEDS = [
    12345, 67890, 11111, 22222, 33333,
    44444, 55555, 66666, 77777, 88888,
    99999, 10203, 40506, 70809, 10111,
    21314, 51617, 81920, 32123, 45678
]


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
        distance = float((observation[i] + 1.0) * 500.0)
        sensors[sensor_names[i]] = distance

        # Also add numbered versions for compatibility
        sensors[f"sensor_{i}"] = distance
        sensors[str(i)] = distance

    # Extract velocity
    vel_x = float(observation[num_sensors] * 30.0)
    vel_y = float(observation[num_sensors + 1] * 10.0)

    return {
        "sensors": sensors,
        "velocity": {"x": vel_x, "y": vel_y},
        "did_crash": False,
    }


def run_single_benchmark(seed: int, verbose: bool = False) -> dict:
    """Run a single benchmark episode with the given seed."""
    
    # Create environment with specific seed
    env = RaceCarEnv(render_mode=None, seed=seed)  # No rendering for benchmarking
    controller = LaneChangeController(verbose=verbose)
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        done = False
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
        
        # Episode complete
        episode_time = time.time() - start_time
        episode_distance = info.get("distance", 0)
        crashed = info.get("crashed", False)
        
        return {
            "seed": seed,
            "distance": episode_distance,
            "reward": episode_reward,
            "steps": episode_steps,
            "crashed": crashed,
            "time": episode_time,
            "action_counts": action_counts
        }
        
    finally:
        controller.close()
        env.close()


def run_benchmark(verbose: bool = False, save_results: bool = True) -> dict:
    """Run the complete benchmark across all seeds."""
    
    print("🏁 Starting Bot with Memory Benchmark")
    print("=" * 50)
    print(f"Running {len(BENCHMARK_SEEDS)} episodes with predefined seeds")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    try:
        for i, seed in enumerate(BENCHMARK_SEEDS):
            print(f"\n🏎️  Episode {i + 1}/{len(BENCHMARK_SEEDS)} (Seed: {seed})")
            print("-" * 40)
            
            try:
                result = run_single_benchmark(seed, verbose=verbose)
                results.append(result)
                
                # Print episode summary
                print(f"  Distance: {result['distance']:.1f}")
                print(f"  Reward: {result['reward']:.2f}")
                print(f"  Steps: {result['steps']}")
                print(f"  Time: {result['time']:.1f}s")
                print(f"  Crashed: {'Yes' if result['crashed'] else 'No'}")
                
            except Exception as e:
                print(f"❌ Error in episode {i + 1} (seed {seed}): {e}")
                results.append({
                    "seed": seed,
                    "distance": 0,
                    "reward": 0,
                    "steps": 0,
                    "crashed": True,
                    "time": 0,
                    "action_counts": {},
                    "error": str(e)
                })
    
    except KeyboardInterrupt:
        print("\n\n🛑 Benchmark interrupted by user")
    
    # Calculate statistics
    total_time = time.time() - start_time
    
    if results:
        distances = [r["distance"] for r in results if "error" not in r]
        rewards = [r["reward"] for r in results if "error" not in r]
        steps = [r["steps"] for r in results if "error" not in r]
        crashes = [r["crashed"] for r in results if "error" not in r]
        times = [r["time"] for r in results if "error" not in r]
        
        # Summary statistics
        summary = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "total_episodes": len(results),
                "successful_episodes": len(distances),
                "failed_episodes": len(results) - len(distances),
                "total_time": total_time,
                "seeds_used": BENCHMARK_SEEDS[:len(results)]
            },
            "performance_stats": {
                "avg_distance": float(np.mean(distances)) if distances else 0,
                "median_distance": float(median(distances)) if distances else 0,
                "std_distance": float(np.std(distances)) if distances else 0,
                "min_distance": float(np.min(distances)) if distances else 0,
                "max_distance": float(np.max(distances)) if distances else 0,
                "avg_reward": float(np.mean(rewards)) if rewards else 0,
                "median_reward": float(median(rewards)) if rewards else 0,
                "std_reward": float(np.std(rewards)) if rewards else 0,
                "avg_steps": float(np.mean(steps)) if steps else 0,
                "median_steps": float(median(steps)) if steps else 0,
                "avg_time": float(np.mean(times)) if times else 0,
                "median_time": float(median(times)) if times else 0,
                "crash_rate": float(sum(crashes) / len(crashes)) if crashes else 1.0
            },
            "episode_results": results
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("📈 Benchmark Results")
        print("=" * 50)
        print(f"Total Episodes: {len(results)}")
        print(f"Successful Episodes: {len(distances)}")
        print(f"Failed Episodes: {len(results) - len(distances)}")
        print(f"Total Time: {total_time:.1f}s")
        print()
        if distances:
            print(f"Distance Stats:")
            print(f"  Average: {np.mean(distances):.1f} ± {np.std(distances):.1f}")
            print(f"  Median: {median(distances):.1f}")
            print(f"  Range: {np.min(distances):.1f} - {np.max(distances):.1f}")
            print()
            print(f"Reward Stats:")
            print(f"  Average: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"  Median: {median(rewards):.2f}")
            print()
            print(f"Steps Stats:")
            print(f"  Average: {np.mean(steps):.0f}")
            print(f"  Median: {median(steps):.0f}")
            print()
            print(f"Time Stats:")
            print(f"  Average Time/Episode: {np.mean(times):.1f}s")
            print(f"  Median Time/Episode: {median(times):.1f}s")
            print()
            print(f"Crash Rate: {sum(crashes) / len(crashes) * 100:.1f}%")
            
            # Best and worst episodes
            best_idx = np.argmax(distances)
            worst_idx = np.argmin(distances)
            print(f"\n🏆 Best Episode: Seed {results[best_idx]['seed']}, Distance {distances[best_idx]:.1f}")
            print(f"🔻 Worst Episode: Seed {results[worst_idx]['seed']}, Distance {distances[worst_idx]:.1f}")
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {filename}")
        
        return summary
    
    else:
        print("\n❌ No results to analyze")
        return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark bot_with_memory performance")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose controller logging")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    print("🎯 Bot with Memory Benchmark")
    print("\nThis benchmark runs the lane change controller across 20 predefined seeds")
    print("for consistent performance evaluation.")
    print("\nPress CTRL+C to stop early\n")
    
    results = run_benchmark(verbose=args.verbose, save_results=not args.no_save)

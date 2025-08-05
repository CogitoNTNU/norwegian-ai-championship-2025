"""
Benchmark bot_with_memory controller in the PPO environment.
This script runs the bot across 200 predefined seeds for consistent testing using multiprocessing.
"""

import sys
import os
import numpy as np
import time
import json
from datetime import datetime
from statistics import median
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add ppo directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "ppo"))

from src.environments.race_car_gym_env import RaceCarEnv
from rule_based_AI.bot_with_memory import LaneChangeController


# Predefined seeds for consistent benchmarking
BENCHMARK_SEEDS = list(range(100, 300))

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
            
            episode_steps += 1
        
        # Episode complete
        episode_time = time.time() - start_time
        episode_distance = info.get("distance", 0)
        crashed = info.get("crashed", False)
        
        return {
            "seed": seed,
            "distance": episode_distance,
            "steps": episode_steps,
            "crashed": crashed,
            "time": episode_time,
            "action_counts": action_counts
        }
        
    except Exception as e:
        return {
            "seed": seed,
            "distance": 0,
            "steps": 0,
            "crashed": True,
            "time": 0,
            "action_counts": {},
            "error": str(e)
        }
        
    finally:
        try:
            controller.close()
        except:
            pass
        try:
            env.close()
        except:
            pass


def run_benchmark_worker(args):
    """Worker function for multiprocessing."""
    seed, verbose = args
    return run_single_benchmark(seed, verbose)


def run_benchmark(verbose: bool = False, save_results: bool = True, num_processes: int = None) -> dict:
    """Run the complete benchmark across all seeds using multiprocessing."""
    
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(BENCHMARK_SEEDS))
    
    print("🏁 Starting Bot with Memory Benchmark (Multi-processed)")
    print("=" * 60)
    print(f"Running {len(BENCHMARK_SEEDS)} episodes with predefined seeds")
    print(f"Using {num_processes} processes")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    completed_count = 0
    
    try:
        # Prepare arguments for worker processes
        worker_args = [(seed, verbose) for seed in BENCHMARK_SEEDS]
        
        # Use ProcessPoolExecutor for better control and progress tracking
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all jobs
            future_to_seed = {
                executor.submit(run_benchmark_worker, args): args[0] 
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Print progress
                    print(f"✅ Episode {completed_count}/{len(BENCHMARK_SEEDS)} complete (Seed: {seed}) - "
                          f"Distance: {result['distance']:.1f}, "
                          f"Crashed: {'Yes' if result['crashed'] else 'No'}")
                    
                except Exception as e:
                    print(f"❌ Error in episode with seed {seed}: {e}")
                    results.append({
                        "seed": seed,
                        "distance": 0,
                        "steps": 0,
                        "crashed": True,
                        "time": 0,
                        "action_counts": {},
                        "error": str(e)
                    })
    
    except KeyboardInterrupt:
        print("\n\n🛑 Benchmark interrupted by user")
        print(f"Completed {completed_count}/{len(BENCHMARK_SEEDS)} episodes before interruption")
    
    # Calculate statistics
    total_time = time.time() - start_time
    
    if results:
        # Sort results by seed for consistent ordering
        results.sort(key=lambda x: x['seed'])
        
        distances = [r["distance"] for r in results if "error" not in r]
        steps = [r["steps"] for r in results if "error" not in r]
        crashes = [r["crashed"] for r in results if "error" not in r]
        times = [r["time"] for r in results if "error" not in r]
        
        # Summary statistics
        summary = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "total_episodes": len(results),
                "successful_episodes": len(distances),
                "num_processes": num_processes,
            },
            "performance_stats": {
                "avg_distance": float(np.mean(distances)) if distances else 0,
                "median_distance": float(median(distances)) if distances else 0,
                "std_distance": float(np.std(distances)) if distances else 0,
                "min_distance": float(np.min(distances)) if distances else 0,
                "max_distance": float(np.max(distances)) if distances else 0,
                "avg_steps": float(np.mean(steps)) if steps else 0,
                "medain_steps": float(median(steps)) if steps else 0,
                "median_steps": float(median(steps)) if steps else 0,
                "min_steps": float(np.min(steps)) if steps else 0,
                "avg_time": float(np.mean(times)) if times else 0,
                "median_time": float(median(times)) if times else 0,
                "crash_rate": float(sum(crashes) / len(crashes)) if crashes else 1.0,
                "episodes_per_second": len(results) / total_time if total_time > 0 else 0
            },
            "episode_results": results
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 Benchmark Results")
        print("=" * 60)
        print(f"Total Episodes: {len(results)}")
        print(f"Successful Episodes: {len(distances)}")
        print(f"Failed Episodes: {len(results) - len(distances)}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Episodes per Second: {len(results) / total_time:.2f}")
        print(f"Processes Used: {num_processes}")
        print()
        if distances:
            print(f"Distance Stats:")
            print(f"  Average: {np.mean(distances):.1f} ± {np.std(distances):.1f}")
            print(f"  Median: {median(distances):.1f}")
            print(f"  Range: {np.min(distances):.1f} - {np.max(distances):.1f}")
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
            if distances:
                best_result = max([r for r in results if "error" not in r], key=lambda x: x['distance'])
                worst_result = min([r for r in results if "error" not in r], key=lambda x: x['distance'])
                print(f"\n🏆 Best Episode: Seed {best_result['seed']}, Distance {best_result['distance']:.1f}")
                print(f"🔻 Worst Episode: Seed {worst_result['seed']}, Distance {worst_result['distance']:.1f}")
        
        # Save results if requested
        if save_results:
            # Create benchmark_results directory if it doesn't exist
            os.makedirs("benchmark_results", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results/benchmark_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n💾 Results saved to: {filename}")
        
        return summary
    
    else:
        print("\n❌ No results to analyze")
        return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark bot_with_memory performance using multiprocessing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose controller logging")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--processes", type=int, default=None, 
                       help="Number of processes to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    print("🎯 Bot with Memory Benchmark (Multi-processed)")
    print("\nThis benchmark runs the lane change controller across 200 predefined seeds")
    print("for consistent performance evaluation using multiple processes.")
    print("\nPress CTRL+C to stop early\n")
    
    results = run_benchmark(
        verbose=args.verbose, 
        save_results=not args.no_save,
        num_processes=args.processes
    )

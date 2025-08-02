from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import numpy as np
import torch
from tqdm import trange
import pygame

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test


# Rainbow DQN for Race Car Environment
parser = argparse.ArgumentParser(description="Rainbow DQN for Race Car")
parser.add_argument("--id", type=str, default="race_car_training", help="Experiment ID")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--T-max",
    type=int,
    default=int(1e6),
    metavar="STEPS",
    help="Number of training steps",
)
parser.add_argument(
    "--max-episode-length",
    type=int,
    default=3600,
    metavar="LENGTH",
    help="Max episode length in steps (3600 = 60 seconds at 60 FPS)",
)
parser.add_argument(
    "--history-length",
    type=int,
    default=4,
    metavar="T",
    help="Number of consecutive states processed",
)
parser.add_argument(
    "--architecture",
    type=str,
    default="race_car",
    choices=["race_car"],
    metavar="ARCH",
    help="Network architecture",
)
parser.add_argument(
    "--hidden-size", type=int, default=512, metavar="SIZE", help="Network hidden size"
)
parser.add_argument(
    "--noisy-std",
    type=float,
    default=0.1,
    metavar="σ",
    help="Initial standard deviation of noisy linear layers",
)
parser.add_argument(
    "--atoms",
    type=int,
    default=51,
    metavar="C",
    help="Discretised size of value distribution",
)
parser.add_argument(
    "--V-min",
    type=float,
    default=-100,
    metavar="V",
    help="Minimum of value distribution support",
)
parser.add_argument(
    "--V-max",
    type=float,
    default=1000,
    metavar="V",
    help="Maximum of value distribution support",
)
parser.add_argument(
    "--model", type=str, metavar="PARAMS", help="Pretrained model (state dict)"
)
parser.add_argument(
    "--memory-capacity",
    type=int,
    default=int(5e4),
    metavar="CAPACITY",
    help="Experience replay memory capacity",
)
parser.add_argument(
    "--replay-frequency",
    type=int,
    default=4,
    metavar="k",
    help="Frequency of sampling from memory",
)
parser.add_argument(
    "--priority-exponent",
    type=float,
    default=0.5,
    metavar="ω",
    help="Prioritised experience replay exponent (originally denoted α)",
)
parser.add_argument(
    "--priority-weight",
    type=float,
    default=0.4,
    metavar="β",
    help="Initial prioritised experience replay importance sampling weight",
)
parser.add_argument(
    "--multi-step",
    type=int,
    default=3,
    metavar="n",
    help="Number of steps for multi-step return",
)
parser.add_argument(
    "--discount", type=float, default=0.99, metavar="γ", help="Discount factor"
)
parser.add_argument(
    "--target-update",
    type=int,
    default=int(1e4),
    metavar="τ",
    help="Number of steps after which to update target network",
)
parser.add_argument(
    "--reward-clip",
    type=int,
    default=0,
    metavar="VALUE",
    help="Reward clipping (0 to disable)",
)
parser.add_argument(
    "--learning-rate", type=float, default=0.0001, metavar="η", help="Learning rate"
)
parser.add_argument(
    "--adam-eps", type=float, default=1.5e-4, metavar="ε", help="Adam epsilon"
)
parser.add_argument(
    "--batch-size", type=int, default=16, metavar="SIZE", help="Batch size"
)
parser.add_argument(
    "--norm-clip",
    type=float,
    default=10,
    metavar="NORM",
    help="Max L2 norm for gradient clipping",
)
parser.add_argument(
    "--learn-start",
    type=int,
    default=int(1e3),
    metavar="STEPS",
    help="Number of steps before starting training",
)
parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
parser.add_argument(
    "--evaluation-interval",
    type=int,
    default=10000,
    metavar="STEPS",
    help="Number of training steps between evaluations",
)
parser.add_argument(
    "--evaluation-episodes",
    type=int,
    default=5,
    metavar="N",
    help="Number of evaluation episodes to average over",
)
parser.add_argument(
    "--evaluation-size",
    type=int,
    default=500,
    metavar="N",
    help="Number of transitions to use for validating Q",
)
parser.add_argument(
    "--render", action="store_true", help="Display screen (testing only)"
)
parser.add_argument(
    "--enable-cudnn",
    action="store_true",
    help="Enable cuDNN (faster but nondeterministic)",
)
parser.add_argument(
    "--checkpoint-interval",
    default=1_000,
    help="How often to checkpoint the model, defaults to 0 (never checkpoint)",
)
parser.add_argument("--memory", help="Path to save/load the memory from")
parser.add_argument(
    "--disable-bzip-memory",
    action="store_true",
    help="Don't zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)",
)
parser.add_argument(
    "--watch",
    action="store_true",
    help="Watch the trained model play with visual rendering",
)
parser.add_argument(
    "--watch-episodes", type=int, default=3, help="Number of episodes to watch"
)


def main(custom_args=None):
    # Setup
    if custom_args is not None:
        args = parser.parse_args(custom_args)
    else:
        args = parser.parse_args()

    print(" " * 26 + "Options")
    for k, v in vars(args).items():
        print(" " * 26 + k + ": " + str(v))
    results_dir = os.path.join(os.path.dirname(__file__), "results", args.id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    metrics = {"steps": [], "rewards": [], "Qs": [], "best_avg_reward": -float("inf")}
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device("cuda")
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device("cpu")

    # Environment
    env = Env(args)
    env.train()
    action_space = env.action_space()

    # Agent
    dqn = Agent(args, env)

    # If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
    if args.model is not None and not args.evaluate:
        if not args.memory:
            raise ValueError(
                "Cannot resume training without memory save path. Aborting..."
            )
        elif not os.path.exists(args.memory):
            raise ValueError(
                "Could not find memory file at {path}. Aborting...".format(
                    path=args.memory
                )
            )

        mem = load_memory(args.memory, args.disable_bzip_memory)

    else:
        mem = ReplayMemory(args, args.memory_capacity)

    priority_weight_increase = (1 - args.priority_weight) / (
        args.T_max - args.learn_start
    )

    # Construct validation memory
    val_mem = ReplayMemory(args, args.evaluation_size)
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            state = env.reset()

        next_state, _, done = env.step(np.random.randint(0, action_space))
        val_mem.append(state, -1, 0.0, done)
        state = next_state
        T += 1

    if args.watch:
        # Watch mode - visual rendering like PPO
        watch_rainbow_model(args)
        return
    elif args.evaluate:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(
            args, 0, dqn, val_mem, metrics, results_dir, evaluate=True
        )  # Test
        print("Avg. reward: " + str(avg_reward) + " | Avg. Q: " + str(avg_Q))
    else:
        # Training loop
        dqn.train()
        done = True
        for T in trange(1, args.T_max + 1):
            if done:
                state = env.reset()

            if T % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

            action = dqn.act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = env.step(action)  # Step
            if args.reward_clip > 0:
                reward = max(
                    min(reward, args.reward_clip), -args.reward_clip
                )  # Clip rewards
            mem.append(state, action, reward, done)  # Append transition to memory

            # Train and test
            if T >= args.learn_start:
                mem.priority_weight = min(
                    mem.priority_weight + priority_weight_increase, 1
                )  # Anneal importance sampling weight β to 1

                if T % args.replay_frequency == 0:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None  # Clear GPU memory

                if T % args.evaluation_interval == 0:
                    dqn.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q = test(
                        args, T, dqn, val_mem, metrics, results_dir
                    )  # Test
                    log(
                        "T = "
                        + str(T)
                        + " / "
                        + str(args.T_max)
                        + " | Avg. reward: "
                        + str(avg_reward)
                        + " | Avg. Q: "
                        + str(avg_Q)
                    )
                    dqn.train()  # Set DQN (online network) back to training mode

                    # If memory path provided, save it
                    if args.memory is not None:
                        save_memory(mem, args.memory, args.disable_bzip_memory)

                # Update target network
                if T % args.target_update == 0:
                    dqn.update_target_net()

                # Checkpoint the network
                if (args.checkpoint_interval != 0) and (
                    T % args.checkpoint_interval == 0
                ):
                    print("Saving rainbow dqn model!")
                    print("T: ", T)
                    dqn.save(results_dir, "checkpoint.pth")

            state = next_state

        dqn.save(results_dir, "model.pth")  # Save final model

    env.close()


# Simple ISO 8601 timestamped logger
def log(s):
    print("[" + str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) + "] " + s)


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, "rb") as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, "wb") as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, "wb") as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


def watch_rainbow_model(args):
    """
    Watch the trained Rainbow DQN model play the race car game with visual rendering.
    """

    # Check if model exists
    results_dir = os.path.join(os.path.dirname(__file__), "results", args.id)
    model_path = os.path.join(results_dir, "checkpoint.pth")

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print(
            f"Please train a model first or check the model path in results/{args.id}/"
        )
        return

    print(f"Loading Rainbow DQN model from {model_path}...")

    # Set up device
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # Environment
    env = Env(args)
    env.train()  # Set environment to use real game logic

    # Agent
    dqn = Agent(args, env)
    dqn.online_net.load_state_dict(torch.load(model_path, map_location=args.device))
    dqn.eval()  # Set to evaluation mode
    print("Model loaded successfully!")

    # Initialize pygame display
    pygame.init()
    screen = pygame.display.set_mode((1600, 1200))
    pygame.display.set_caption("Rainbow DQN Race Car Agent")
    clock = pygame.time.Clock()

    print(f"\nWatching Rainbow DQN model for {args.watch_episodes} episodes...")
    print("Press ESC to quit, SPACE to pause")

    for episode in range(args.watch_episodes):
        print(f"\n=== Episode {episode + 1}/{args.watch_episodes} ===")

        state = env.reset()
        episode_reward = 0
        step_count = 0
        paused = False
        done = False

        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")

            if not paused:
                # Get action from Rainbow DQN model
                action = dqn.act(state)

                # Take step in environment
                next_state, reward, done = env.step(action)
                episode_reward += reward
                step_count += 1

                # Print progress every 100 steps (every ~1.67 seconds)
                if step_count % 100 == 0:
                    time_seconds = step_count / 60.0
                    print(
                        f"  Step {step_count} ({time_seconds:.1f}s/{60}s): Reward = {episode_reward:.2f}"
                    )

                # Check if episode is done
                if done:
                    time_seconds = step_count / 60.0
                    print(f"  Episode finished: Reward = {episode_reward:.2f}")
                    print(
                        f"  Result: Episode completed at step {step_count} ({time_seconds:.1f}s)"
                    )
                    break

                state = next_state

            # Render the game using game state
            screen.fill((0, 0, 0))

            # Import STATE from the actual game
            try:
                import src.game.core as core

                STATE = core.STATE

                # Draw road
                if hasattr(STATE, "road") and STATE.road:
                    if hasattr(STATE.road, "surface") and STATE.road.surface:
                        screen.blit(STATE.road.surface, (0, 0))

                    # Draw walls
                    if hasattr(STATE.road, "walls"):
                        for wall in STATE.road.walls:
                            if hasattr(wall, "draw"):
                                wall.draw(screen)

                # Draw cars
                if hasattr(STATE, "cars") and STATE.cars:
                    for car in STATE.cars:
                        if hasattr(car, "sprite") and car.sprite:
                            screen.blit(car.sprite, (car.x, car.y))
                            # Draw bounding box
                            bounds = car.get_bounds()
                            color = (255, 255, 0) if car == STATE.ego else (255, 0, 0)
                            pygame.draw.rect(screen, color, bounds, width=2)

                # Draw sensors
                if hasattr(STATE, "sensors") and STATE.sensors:
                    for sensor in STATE.sensors:
                        if hasattr(sensor, "draw"):
                            sensor.draw(screen)

            except ImportError:
                # Fallback if we can't import the game state
                pass

            # Draw HUD info
            font = pygame.font.Font(None, 36)
            time_seconds = step_count / 60.0
            texts = [
                f"Episode: {episode + 1}/{args.watch_episodes}",
                f"Time: {time_seconds:.1f}s / 60s",
                f"Step: {step_count} / 3600",
                f"Reward: {episode_reward:.1f}",
                "",
                "Rainbow DQN Agent",
                "Controls:",
                "ESC - Quit",
                "SPACE - Pause/Resume",
            ]

            for i, text in enumerate(texts):
                if text:  # Skip empty lines
                    if text == "Rainbow DQN Agent":
                        color = (0, 255, 255)  # Cyan for Rainbow
                    elif text.startswith("Controls"):
                        color = (255, 255, 0)  # Yellow for controls
                    else:
                        color = (255, 255, 255)  # White for normal text

                    text_surface = font.render(text, True, color)
                    screen.blit(text_surface, (10, 10 + i * 30))

            # Show pause indicator
            if paused:
                pause_font = pygame.font.Font(None, 72)
                pause_text = pause_font.render("PAUSED", True, (255, 0, 0))
                screen.blit(
                    pause_text,
                    (screen.get_width() // 2 - 100, screen.get_height() // 2),
                )

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

    print("\nDemo complete!")
    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()

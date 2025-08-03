"""
Main entry point for Race Car AI Training and Evaluation
Supports both PPO and Rainbow DQN algorithms
"""

import argparse
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Race Car AI Training and Evaluation")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "rainbow"],
        required=True,
        help="Choose the RL algorithm to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "watch"],
        required=True,
        help="Choose to train a new model or watch an existing one",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for watching (required for watch mode)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(1e9),
        help="Number of training timesteps (for train mode)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb logging (PPO only)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without rendering (default: True)",
    )

    # Rainbow specific arguments
    parser.add_argument(
        "--rainbow-episodes",
        type=int,
        default=5,
        help="Number of episodes for Rainbow evaluation",
    )

    args = parser.parse_args()

    print(f"🏁 Race Car AI - {args.algorithm.upper()} Algorithm")
    print(f"Mode: {args.mode}")
    print("=" * 50)

    if args.algorithm == "ppo":
        run_ppo(args)
    elif args.algorithm == "rainbow":
        run_rainbow(args)


def run_ppo(args):
    """Run PPO algorithm"""
    if args.mode == "train":
        print("🚀 Starting PPO Training...")

        # Import and run PPO training
        sys.path.append(os.path.join(os.path.dirname(__file__), "ppo"))
        from train_ppo_real import train_real_ppo_model

        train_real_ppo_model(timesteps=args.timesteps, use_wandb=not args.no_wandb)
        print("✅ PPO Training completed! Model saved.")

    elif args.mode == "watch":
        print("👀 Watching PPO Model...")

        if not args.model_path:
            # Try to find the latest model
            model_files = (
                list(Path("models").glob("*.zip")) if Path("models").exists() else []
            )
            if model_files:
                args.model_path = str(max(model_files, key=os.path.getctime))
                print(f"🔍 Auto-detected model: {args.model_path}")
            else:
                print("❌ No model specified and no models found in ./models/")
                print("Use --model-path to specify a model file")
                return

        # Import and run PPO watching
        sys.path.append(os.path.join(os.path.dirname(__file__), "ppo"))
        from watch_ppo_real import watch_ppo_model

        watch_ppo_model(args.model_path, headless=args.headless)


def run_rainbow(args):
    """Run Rainbow DQN algorithm"""
    if args.mode == "train":
        print("🌈 Starting Rainbow DQN Training...")

        # Setup Rainbow training
        rainbow_dir = os.path.join(os.path.dirname(__file__), "Rainbow")

        # Add Rainbow directory to path and import
        sys.path.insert(0, rainbow_dir)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rainbow_main", os.path.join(rainbow_dir, "main.py")
        )
        rainbow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rainbow_module)

        cmd_args = [
            "--T-max",
            str(args.timesteps),
            "--evaluation-episodes",
            str(args.rainbow_episodes),
            "--learn-start",
            "1000",
            "--evaluation-interval",
            str(args.timesteps // 10),
            "--id",
            "race_car_training",
        ]

        if args.headless:
            cmd_args.append("--render")

        try:
            # This will run the Rainbow main function
            rainbow_module.main(cmd_args)
        except SystemExit:
            pass

        print("✅ Rainbow Training completed!")

    elif args.mode == "watch":
        print("👀 Watching Rainbow Model...")

        if not args.model_path:
            # Try to find the latest Rainbow model
            rainbow_dir = os.path.join(os.path.dirname(__file__), "Rainbow", "results")
            if os.path.exists(rainbow_dir):
                model_files = []
                for subdir in os.listdir(rainbow_dir):
                    model_path = os.path.join(rainbow_dir, subdir, "model.pth")
                    if os.path.exists(model_path):
                        model_files.append(model_path)

                if model_files:
                    args.model_path = max(model_files, key=os.path.getctime)
                    print(f"🔍 Auto-detected model: {args.model_path}")
                else:
                    print("❌ No Rainbow model found in results directory")
                    return
            else:
                print("❌ No model specified and no Rainbow results found")
                print("Use --model-path to specify a model file")
                return

        # Setup Rainbow evaluation
        rainbow_dir = os.path.join(os.path.dirname(__file__), "Rainbow")

        # Add Rainbow directory to path and import
        sys.path.insert(0, rainbow_dir)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rainbow_main", os.path.join(rainbow_dir, "main.py")
        )
        rainbow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rainbow_module)

        cmd_args = [
            "--model",
            args.model_path,
            "--evaluate",
            "--evaluation-episodes",
            str(args.rainbow_episodes),
        ]

        cmd_args.append("--watch")

        try:
            # This will run the Rainbow evaluation
            rainbow_module.main(cmd_args)
        except SystemExit:
            pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Utility script to manage PPO model checkpoints.
"""

import os
import glob
from datetime import datetime
import argparse


def list_checkpoints():
    """List all available checkpoints."""
    checkpoints = glob.glob("models/checkpoint_*.zip")
    if not checkpoints:
        print("No checkpoints found in models/ directory")
        return

    print("Available checkpoints:")
    print("-" * 50)
    for checkpoint in sorted(checkpoints):
        # Extract timestamp from filename
        basename = os.path.basename(checkpoint)
        timestamp_str = basename.replace("checkpoint_", "").replace(".zip", "")

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Get file size
            size_mb = os.path.getsize(checkpoint) / (1024 * 1024)

            print(f"  {basename:<30} | {formatted_time} | {size_mb:.1f} MB")
        except ValueError:
            print(f"  {basename:<30} | Invalid timestamp format")


def get_latest_checkpoint():
    """Get the path to the most recent checkpoint."""
    checkpoints = glob.glob("models/checkpoint_*.zip")
    if not checkpoints:
        return None

    latest = max(checkpoints, key=os.path.getctime)
    return latest.replace(".zip", "")


def clean_old_checkpoints(keep_count=5):
    """Keep only the N most recent checkpoints."""
    checkpoints = glob.glob("models/checkpoint_*.zip")
    if len(checkpoints) <= keep_count:
        print(f"Only {len(checkpoints)} checkpoints found, keeping all")
        return

    # Sort by creation time (oldest first)
    checkpoints_sorted = sorted(checkpoints, key=os.path.getctime)
    to_delete = checkpoints_sorted[:-keep_count]

    print(
        f"Deleting {len(to_delete)} old checkpoints, keeping {keep_count} most recent:"
    )
    for checkpoint in to_delete:
        print(f"  Deleting: {os.path.basename(checkpoint)}")
        os.remove(checkpoint)

    print("Cleanup completed!")


def main():
    parser = argparse.ArgumentParser(description="Manage PPO model checkpoints")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument(
        "--latest", action="store_true", help="Show path to latest checkpoint"
    )
    parser.add_argument(
        "--clean", type=int, metavar="N", help="Keep only N most recent checkpoints"
    )

    args = parser.parse_args()

    if args.list:
        list_checkpoints()
    elif args.latest:
        latest = get_latest_checkpoint()
        if latest:
            print(f"Latest checkpoint: {latest}")
            print(
                f"To resume training: python train_ppo_real.py --resume-from {latest}"
            )
        else:
            print("No checkpoints found")
    elif args.clean is not None:
        clean_old_checkpoints(args.clean)
    else:
        print("PPO Checkpoint Manager")
        print("=" * 30)
        list_checkpoints()
        print("\nUsage examples:")
        print(
            "  python manage_checkpoints.py --list                    # List all checkpoints"
        )
        print(
            "  python manage_checkpoints.py --latest                  # Show latest checkpoint"
        )
        print(
            "  python manage_checkpoints.py --clean 3                 # Keep only 3 newest"
        )
        print(
            "  python train_ppo_real.py --resume-from models/checkpoint_20250801_143309"
        )


if __name__ == "__main__":
    main()

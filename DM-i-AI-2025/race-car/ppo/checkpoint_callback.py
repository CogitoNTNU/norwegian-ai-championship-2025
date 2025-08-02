"""
Checkpoint callback for saving models during training.
"""

import os
from stable_baselines3.common.callbacks import BaseCallback
import wandb


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints during training.

    :param save_freq: Save model every save_freq steps
    :param save_path: Path to save models
    :param name_prefix: Prefix for model names
    :param save_to_wandb: Whether to upload models to W&B
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "checkpoint",
        save_to_wandb: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_to_wandb = save_to_wandb
        self.saved_models = []

    def _init_callback(self) -> None:
        # Create save directory if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            self.saved_models.append(path + ".zip")

            if self.verbose > 0:
                print(f"\nSaved checkpoint at {self.num_timesteps} steps to {path}.zip")

            # Upload to W&B if enabled
            if self.save_to_wandb and wandb.run is not None:
                try:
                    # Use policy="now" to avoid symlink issues on Windows
                    wandb.save(path + ".zip", base_path=self.save_path, policy="now")

                    # Log metrics about the checkpoint
                    wandb.log(
                        {
                            "checkpoint/timesteps": self.num_timesteps,
                            "checkpoint/episodes": len(self.model.ep_info_buffer),
                            "checkpoint/saved": 1,
                        },
                        step=self.num_timesteps,
                    )

                    if self.verbose > 0:
                        print("   Uploaded to W&B")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"   Warning: Failed to upload to W&B: {e}")
                        print("   Checkpoint saved locally")

        return True

    def _on_training_end(self) -> None:
        """Save final model and summary."""
        # Save final model
        final_path = os.path.join(self.save_path, f"{self.name_prefix}_final")
        self.model.save(final_path)

        if self.verbose > 0:
            print(f"\nSaved final model to {final_path}.zip")
            print(f"   Total checkpoints saved: {len(self.saved_models) + 1}")

        # Upload final model to W&B
        if self.save_to_wandb and wandb.run is not None:
            try:
                # Use policy="now" to avoid symlink issues on Windows
                wandb.save(final_path + ".zip", base_path=self.save_path, policy="now")

                # Save list of all checkpoints
                checkpoint_list = {
                    "checkpoints": self.saved_models + [final_path + ".zip"],
                    "total_timesteps": self.num_timesteps,
                    "save_freq": self.save_freq,
                }
                wandb.log({"checkpoint_summary": checkpoint_list})

                if self.verbose > 0:
                    print("   Final model uploaded to W&B")
            except Exception as e:
                if self.verbose > 0:
                    print(f"   Warning: Failed to upload final model to W&B: {e}")
                    print("   Final model saved locally")

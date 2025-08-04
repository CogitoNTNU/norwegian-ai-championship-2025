import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from race_car_gym_env import RaceCarEnv
from datetime import datetime
import json
import wandb


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs):
        shared_features = self.shared(obs)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value

    def get_action(self, obs):
        action_probs, value = self.forward(obs)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value


class PPOTrainer:
    """PPO training algorithm."""

    def __init__(self, env, config, use_wandb=True):
        self.env = env
        self.config = config
        self.use_wandb = use_wandb

        # Get dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Initialize network
        self.policy = PPONetwork(self.obs_dim, self.action_dim, config["hidden_dim"])
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["learning_rate"]
        )

        # Training parameters
        self.gamma = config["gamma"]
        self.lambda_gae = config["lambda_gae"]
        self.epsilon = config["epsilon"]
        self.value_loss_coef = config["value_loss_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]

        # Logging
        self.writer = SummaryWriter(
            f"runs/ppo_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.episode_count = 0
        self.total_steps = 0

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="race-car-ppo-clean",
                config=config,
                tags=["ppo", "race-car", "clean-implementation"],
            )

    def collect_trajectories(self, num_steps):
        """Collect trajectories for training."""
        obs_batch = []
        action_batch = []
        log_prob_batch = []
        value_batch = []
        reward_batch = []
        done_batch = []

        obs, _ = self.env.reset()

        for _ in range(num_steps):
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Get action
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_tensor)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store experience
            obs_batch.append(obs)
            action_batch.append(action)
            log_prob_batch.append(log_prob.item())
            value_batch.append(value.item())
            reward_batch.append(reward)
            done_batch.append(done)

            self.total_steps += 1

            # Log step rewards to wandb
            if self.use_wandb and "reward_breakdown" in info:
                step_logs = {
                    f"Step/{k}": v for k, v in info["reward_breakdown"].items()
                }
                step_logs["Step/TotalReward"] = reward
                wandb.log(step_logs)

            if done:
                # Log episode metrics
                self.log_episode(info)
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Get bootstrap value for the final state if not done
        with torch.no_grad():
            final_obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            _, bootstrap_value = self.policy.forward(final_obs_tensor)
            bootstrap_value = bootstrap_value.item()

        # Convert to tensors
        obs_batch = torch.FloatTensor(np.array(obs_batch))
        action_batch = torch.LongTensor(action_batch)
        log_prob_batch = torch.FloatTensor(log_prob_batch)
        value_batch = torch.FloatTensor(value_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            reward_batch, value_batch, done_batch, bootstrap_value
        )

        return {
            "obs": obs_batch,
            "actions": action_batch,
            "log_probs": log_prob_batch,
            "values": value_batch,
            "advantages": advantages,
            "returns": returns,
        }

    def compute_gae(self, rewards, values, dones, bootstrap_value):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_value = bootstrap_value  # Start with bootstrap value

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0  # Terminal state has no next value
                gae = 0  # Reset GAE for new episode

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(self, trajectories, epochs):
        """Update policy using PPO with minibatches."""
        obs = trajectories["obs"]
        actions = trajectories["actions"]
        old_log_probs = trajectories["log_probs"]
        advantages = trajectories["advantages"]
        returns = trajectories["returns"]

        batch_size = obs.shape[0]
        minibatch_size = min(512, batch_size)  # Use smaller minibatches for stability

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0

        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Get current policy outputs
                action_probs, values = self.policy(mb_obs)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Compute ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), mb_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Accumulate for logging
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        # Average losses for logging
        avg_policy_loss = total_policy_loss / update_count
        avg_value_loss = total_value_loss / update_count
        avg_entropy = total_entropy / update_count

        # Log losses
        self.writer.add_scalar("Loss/Policy", avg_policy_loss, self.total_steps)
        self.writer.add_scalar("Loss/Value", avg_value_loss, self.total_steps)
        self.writer.add_scalar("Loss/Entropy", avg_entropy, self.total_steps)

        # Log to wandb
        if self.use_wandb:
            # Compute additional metrics for debugging
            with torch.no_grad():
                # Compute approximate KL divergence
                action_probs, _ = self.policy(obs)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                approx_kl = (old_log_probs - new_log_probs).mean()

                # Compute policy ratio statistics
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                clip_fraction = (ratio != clipped_ratio).float().mean()

            wandb.log(
                {
                    "Loss/Policy": avg_policy_loss,
                    "Loss/Value": avg_value_loss,
                    "Loss/Entropy": avg_entropy,
                    "Training/LearningRate": self.config["learning_rate"],
                    "Training/ClipRange": self.epsilon,
                    "Training/Steps": self.total_steps,
                    "Training/ApproxKL": approx_kl.item(),
                    "Training/ClipFraction": clip_fraction.item(),
                    "Training/AdvantagesMean": advantages.mean().item(),
                    "Training/AdvantagesStd": advantages.std().item(),
                }
            )

    def log_episode(self, info):
        """Log episode metrics."""
        self.episode_count += 1
        self.writer.add_scalar("Episode/Distance", info["distance"], self.episode_count)
        self.writer.add_scalar(
            "Episode/Reward", info["episode_reward"], self.episode_count
        )
        self.writer.add_scalar(
            "Episode/VelocityX", info["velocity_x"], self.episode_count
        )
        self.writer.add_scalar(
            "Episode/Crashed", float(info["crashed"]), self.episode_count
        )

        # Log to wandb
        if self.use_wandb:
            wandb.log(
                {
                    "Episode/Distance": info["distance"],
                    "Episode/Reward": info["episode_reward"],
                    "Episode/VelocityX": info["velocity_x"],
                    "Episode/VelocityY": info["velocity_y"],
                    "Episode/Crashed": float(info["crashed"]),
                    "Episode/Length": info["ticks"],
                    "Episode/YPosition": info["y_position"],
                    "Episode/Count": self.episode_count,
                }
            )

        print(
            f"Episode {self.episode_count}: Distance={info['distance']:.1f}, "
            f"Reward={info['episode_reward']:.1f}, Crashed={info['crashed']}"
        )

    def train(self, total_timesteps):
        """Main training loop."""
        steps_per_update = self.config["steps_per_update"]
        epochs_per_update = self.config["epochs_per_update"]

        num_updates = total_timesteps // steps_per_update

        for update in range(num_updates):
            # Collect trajectories
            trajectories = self.collect_trajectories(steps_per_update)

            # Update policy
            self.update_policy(trajectories, epochs_per_update)

            # Save model periodically
            if (update + 1) % 10 == 0:
                self.save_model(f"ppo_racecar_update_{update + 1}.pth")

            print(f"Update {update + 1}/{num_updates}, Total Steps: {self.total_steps}")

    def save_model(self, filename):
        """Save model checkpoint."""
        os.makedirs("models", exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episode_count": self.episode_count,
                "total_steps": self.total_steps,
                "config": self.config,
            },
            os.path.join("models", filename),
        )
        print(f"Model saved to models/{filename}")

    def load_model(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint["episode_count"]
        self.total_steps = checkpoint["total_steps"]
        print(f"Model loaded from {filepath}")


def main():
    # Configuration
    config = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "lambda_gae": 0.95,
        "epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "hidden_dim": 256,
        "steps_per_update": 2048,
        "epochs_per_update": 10,
        "total_timesteps": 1_000_000,
    }

    # Save config
    with open("ppo_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create environment
    env = RaceCarEnv(render_mode=None)  # Set to "human" to visualize

    # Create trainer with wandb enabled
    trainer = PPOTrainer(env, config, use_wandb=True)

    # Train
    print("Starting PPO training...")
    trainer.train(config["total_timesteps"])

    # Save final model
    trainer.save_model("ppo_racecar_final.pth")

    # Finish wandb logging
    if trainer.use_wandb:
        wandb.finish()

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()

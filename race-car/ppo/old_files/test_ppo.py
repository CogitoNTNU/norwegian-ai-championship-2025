from race_car_gym_env import RaceCarEnv
from train_ppo_clean import PPOTrainer


def test_ppo_basic_functionality():
    """Test basic PPO functionality without training."""
    print("Testing PPO basic functionality...")

    # Test config
    config = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "lambda_gae": 0.95,
        "epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "hidden_dim": 64,  # Smaller for testing
        "steps_per_update": 128,  # Smaller for testing
        "epochs_per_update": 2,
        "total_timesteps": 1000,
    }

    # Create environment
    env = RaceCarEnv(render_mode=None)
    print(f"Environment created. Obs space: {env.observation_space.shape}")

    # Create trainer
    trainer = PPOTrainer(env, config, use_wandb=False)
    print("PPO trainer created successfully")

    # Test trajectory collection
    print("\nTesting trajectory collection...")
    trajectories = trainer.collect_trajectories(config["steps_per_update"])

    print("Collected trajectories:")
    print(f"  Observations shape: {trajectories['obs'].shape}")
    print(f"  Actions shape: {trajectories['actions'].shape}")
    print(f"  Advantages shape: {trajectories['advantages'].shape}")
    print(f"  Returns shape: {trajectories['returns'].shape}")

    # Test policy update
    print("\nTesting policy update...")
    trainer.update_policy(trajectories, config["epochs_per_update"])
    print("Policy update completed successfully")

    # Test a few training steps
    print("\nTesting short training run...")
    trainer.train(500)  # Just 500 steps

    env.close()
    print("\nAll tests passed! [SUCCESS]")


if __name__ == "__main__":
    test_ppo_basic_functionality()

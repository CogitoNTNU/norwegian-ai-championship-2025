from rule_based_AI.benchmanrk_bot import run_benchmark
from rule_based_AI.watch_bot_with_memory import watch_bot_in_ppo_env

def main():
    print("\n🤖 Bot Control Menu")
    print("=" * 30)
    print("1. Run Benchmark")
    print("2. Watch Bot in PPO Environment")
    print("3. Exit")
    print("=" * 30)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n🏃 Starting benchmark...")
                run_benchmark()
                break
                
            elif choice == "2":
                print("\n👁️ Starting bot watching in PPO environment...")
                watch_bot_in_ppo_env()
                break
                
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()

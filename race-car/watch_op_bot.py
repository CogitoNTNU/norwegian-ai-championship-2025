from bot_with_memory import LaneChangeController
from ppo.race_car_gym_env import RaceCarEnv


def watch():
    ai = LaneChangeController()
    env = RaceCarEnv(render_mode="human")
    obs = env.reset()

    print(obs)
    while True:
        print(obs)
        observation, reward, done, info = env.step(ai.predict_actions(obs))
        env.render()
        print(obs)

        if done:
            print("Episode finished")
            break


if __name__ == "__main__":
    watch()

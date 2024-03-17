import gymnasium as gym
import stable_baselines3 as sb3
import time
import bh_dyn as bh
from gymnasium.wrappers.record_video import RecordVideo

import os

videodir = "./videos/BhDyn/"
modeldir = "./models/BhDyn/"

train_num = ""
USE_Algorithm = None
i = "1"

VIDEO_FLAG = False
STEP_TRI_NUM = 5000
EPI_TRI_NUM = 500
MAX_EPI_STEP = 3000


def main():
    env = gym.make("bh_dyn/BhDyn-v0", render_mode="rgb_array")
    path = os.path.join(modeldir, USE_Algorithm, train_num, i, ".zip")
    model = sb3.SAC.load(path, env)
    Test(model)


def Test(model):
    if VIDEO_FLAG:
        env = gym.make(
            "bh_dyn/BhDyn-v0",
            render_mode="rgb_array",
            max_episode_steps=MAX_EPI_STEP,
        )
        path = os.path.join(videodir, "Test", USE_Algorithm, train_num)
        env = RecordVideo(env, path, episode_trigger=EpiTrigger)
    else:
        env = gym.make(
            "bh_dyn/BhDyn-v0",
            render_mode="human",
            max_episode_steps=MAX_EPI_STEP,
        )

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:

        action, _obs = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward
        done = done1 or done2
        time.sleep(0.1)
        print(f"Total Reward: {total_reward}")

    env.close()


def StepTrigger(step):
    return step % STEP_TRI_NUM == 0


def EpiTrigger(epi):
    return epi % EPI_TRI_NUM == 0


if __name__ == "__main__":
    train_num = input("input train_number\n")
    USE_Algorithm = input("input algorithm you want to test\n")
    main()

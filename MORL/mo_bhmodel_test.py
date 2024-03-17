import mo_gymnasium as mo_gym
import numpy as np
import mo_bh_dyn

# from morl_baselines.multi_policy.capql.capql import CAPQL
from our_capql.capql import CAPQL
from gymnasium.wrappers.record_video import RecordVideo
import os
import wandb

GAMMA = 0.99

logdir = "./logs/BhDyn/"
videodir = "./videos/BhDyn/"
modeldir = "./models/BhDyn/"

train_num = ""
model_name = ""

VIDEO_FLAG = True
STEP_TRI_NUM = 5000
EPI_TRI_NUM = 100
MAX_EPI_STEP = 3000


def main():
    env = mo_gym.make("mo-bh_dyn-v0", render_mode="human")

    model = CAPQL(
        env=env,
        gamma=GAMMA,
        log=False,  # use weights and biases to see the results!
        learning_starts=1000,
        buffer_size=2000000,
        batch_size=256,
        alpha=0.005,
        wandb_entity="207185099",
        # device="cuda",
        seed=3407,
        logdir=logdir,
        train_num=train_num,
    )
    if model_name != "":
        model.load(model_name)
        print(f"Load model {model_name} Successfully!")

    Test(model)


def Test(model):
    env = mo_gym.make("mo-humanoid-v4", render_mode="human")

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = model.eval(obs, np.array([1, 1]))
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward
        done = done1 or done2
        print(f"Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    train_num = input("input train_number\n")
    model_name = input(
        "type your model name\nif don't load model, input enter:\n"
    )
    main()

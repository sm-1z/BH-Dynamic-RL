import mo_gymnasium as mo_gym
import numpy as np
import mo_bh_dyn

# from morl_baselines.multi_policy.capql.capql import CAPQL
from our_capql.capql import CAPQL
from gymnasium.wrappers.record_video import RecordVideo
import os
import wandb

# wandb settings
# os.environ["WANDB_API_KEY"] = "your own WANDB_API_KEY"
# os.environ["WANDB_MODE"] = "offline"

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
    model = Train()


def Train():

    if VIDEO_FLAG:
        env = mo_gym.make("mo-bh_dyn-v0", render_mode="rgb_array")
        path = os.path.join(videodir, "CAPQL", train_num)
        env = RecordVideo(env, path, episode_trigger=EpiTrigger)
    else:
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

    # wandb.init()

    for i in range(40):
        print("Running iteration:", i)
        model.train(
            total_timesteps=50000,
            eval_env=env,
            ref_point=np.array([10, 2, 20, 5]),
            # num_eval_episodes_for_front=5,
            iteration=i,
            checkpoints=False,
        )
        path = os.path.join(modeldir, "CAPQL", train_num)
        model.save(path, filename=str(i), save_replay_buffer=True)
        print("Finish running iteration:", i, "save model")
    return model


def EpiTrigger(epi):
    return epi % EPI_TRI_NUM == 0


if __name__ == "__main__":
    train_num = input("input train_number\n")
    model_name = input(
        "type your model name\nif don't load model, input enter:\n"
    )
    main()

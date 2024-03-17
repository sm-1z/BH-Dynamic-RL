import gymnasium as gym
import stable_baselines3 as sb3
from gymnasium.wrappers.record_video import RecordVideo
import bh_dyn as bh

import os


logdir = "./logs/BhDyn/"
videodir = "./videos/BhDyn/"
modeldir = "./models/BhDyn/"

train_num = ""
USE_Algorithm = None

VIDEO_FLAG = True
STEP_TRI_NUM = 5000
EPI_TRI_NUM = 500
MAX_EPI_STEP = 3000


def main():
    if USE_Algorithm == "TD3":
        model = Train("TD3")
    elif USE_Algorithm == "A2C":
        model = Train("A2C")
    elif USE_Algorithm == "SAC":
        model = Train("SAC")
    else:
        raise Exception("Not have this algorithm！")


def Train(method):
    # 创建环境
    if VIDEO_FLAG:
        env = gym.make(
            "bh_dyn/BhDyn-v0",
            render_mode="rgb_array",
            max_episode_steps=MAX_EPI_STEP,
        )
    else:
        env = gym.make(
            "bh_dyn/BhDyn-v0",
            render_mode="human",
            max_episode_steps=MAX_EPI_STEP,
        )
    if method == "TD3":
        # 视频记录
        if VIDEO_FLAG:
            path = os.path.join(videodir, "TD3", train_num)
            env = RecordVideo(env, path, episode_trigger=EpiTrigger)
        # 创建算法模型
        model = sb3.TD3(
            "MlpPolicy",
            env,
            learning_starts=10000,
            train_freq=1,
            gradient_steps=1,
            learning_rate=1e-3,
            batch_size=256,
            policy_kwargs=dict(net_arch=[400, 300]),
            verbose=1,
            tensorboard_log=os.path.join(logdir, "TD3", train_num),
        )

        for i in range(40):
            model.learn(total_timesteps=50000)
            path = os.path.join(modeldir, "TD3", train_num, str(i))
            # 保存模型
            model.save(path)

    elif method == "A2C":
        if VIDEO_FLAG:
            path = os.path.join(videodir, "A2C", train_num)
            env = RecordVideo(env, path, episode_trigger=EpiTrigger)

        model = sb3.A2C(
            "MlpPolicy",
            env,
            normalize_advantage=True,
            verbose=1,
            tensorboard_log=os.path.join(logdir, "A2C", train_num),
        )
        for i in range(40):
            model.learn(total_timesteps=50000)
            path = os.path.join(modeldir, "A2C", train_num, str(i))
            model.save(path)

    elif method == "SAC":
        if VIDEO_FLAG:
            path = os.path.join(videodir, "SAC", train_num)
            env = RecordVideo(env, path, episode_trigger=EpiTrigger)

        model = sb3.SAC(
            "MlpPolicy",
            env,
            learning_starts=10000,
            verbose=1,
            tensorboard_log=os.path.join(logdir, "SAC", train_num),
        )
        for i in range(40):
            model.learn(total_timesteps=5000)
            path = os.path.join(modeldir, "SAC", train_num, str(i))
            model.save(path)

    return model


def StepTrigger(step):
    return step % STEP_TRI_NUM == 0


def EpiTrigger(epi):
    return epi % EPI_TRI_NUM == 0


def CheckDir():
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(videodir):
        os.makedirs(videodir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)


if __name__ == "__main__":
    CheckDir()
    train_num = input("input train_number\n")
    USE_Algorithm = input("input your favorite algorithm\n")
    main()

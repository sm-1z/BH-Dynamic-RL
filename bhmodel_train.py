import gymnasium as gym
import stable_baselines3 as sb3
import time
import bh_dyn as bh
from gymnasium.wrappers.record_video import RecordVideo

train_num = "1"

logdir = "./logs/"
videodir = "./videos/BH/"

VIDEO_FLAG = True
USE_Algorithm = True


def main():
    if USE_Algorithm == "DDPG":
        model = Train("DDPG")
    elif USE_Algorithm == "A2C":
        model = Train("A2C")
    elif USE_Algorithm == "SAC":
        model = Train("SAC")
    else:
        raise Exception("Not have this algorithm！")
    # env = gym.make("bh_dyn/BhDyn-v0", render_mode="human")
    # model = sb3.SAC.load("./BH/SAC_model309_4", env)
    # Test(model)


def Train(method):
    # 创建环境
    if VIDEO_FLAG:
        env = gym.make(
            "bh_dyn/BhDyn-v0", render_mode="rgb_array", max_episode_steps=3000
        )
    else:
        env = gym.make(
            "bh_dyn/BhDyn-v0", render_mode="human", max_episode_steps=3000
        )

    if method == "DDPG":
        # 视频记录
        if VIDEO_FLAG:
            env = RecordVideo(
                env, videodir + "DDPG_" + train_num, episode_trigger=EpiTrigger
            )
        # 创建算法模型
        model = sb3.DDPG(
            "MlpPolicy",
            env,
            learning_starts=10000,
            train_freq=1,
            gradient_steps=1,
            learning_rate=1e-3,
            batch_size=256,
            policy_kwargs=dict(net_arch=[400, 300]),
            verbose=1,
            tensorboard_log=logdir + "BH/DDPG/" + train_num,
        )

        for i in range(40):
            model.learn(total_timesteps=50000)

            model.save("./BH/DDPG_model" + train_num + "_" + str(i))

    elif method == "A2C":
        if VIDEO_FLAG:
            env = RecordVideo(
                env, videodir + "A2C" + train_num, episode_trigger=EpiTrigger
            )
        model = sb3.A2C(
            "MlpPolicy",
            env,
            normalize_advantage=True,
            verbose=1,
            tensorboard_log=logdir + "BH/A2C/" + train_num,
        )
        # total_timesteps=1e6
        # model = sb3.A2C.load("./BH/A2C_model1250001", env)
        for i in range(40):
            model.learn(total_timesteps=50000)

            model.save("./BH/A2C_model" + train_num + "_" + str(i))

    elif method == "SAC":
        if VIDEO_FLAG:
            # env = RecordVideo(
            #     env, videodir + "SAC" + train_num, step_trigger=StepTrigger
            # )
            env = RecordVideo(
                env, videodir + "SAC" + train_num, episode_trigger=EpiTrigger
            )
        model = sb3.SAC(
            "MlpPolicy",
            env,
            learning_starts=10000,
            verbose=1,
            tensorboard_log=logdir + "BH/SAC" + train_num,
        )
        # total_timesteps=1e6
        # model = sb3.SAC.load("./BH/SAC_model309_4", env)
        for i in range(40):
            model.learn(total_timesteps=50000)

            model.save("./BH/SAC_model" + train_num + "_" + str(i))

    return model


def Test(model):
    # if VIDEO_FLAG:
    #     env = gym.make(
    #         "bh_dyn/BhDyn-v0", render_mode="rgb_array", max_episode_steps=3000
    #     )
    #     # AL = "DDPG" if USE_DDPG else "A2C"
    #     # env = RecordVideo(
    #     #     env,
    #     #     videodir + AL + "Test" + train_num,
    #     # )  # 不需要env.close()
    # else:
    env = gym.make(
        "bh_dyn/BhDyn-v0", render_mode="human", max_episode_steps=3000
    )

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:

        action, _obs = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward
        done = done1 or done2
        # time.sleep(0.1)
        print(f"Total Reward: {total_reward}")

    env.close()


def StepTrigger(step):
    return step % 5000 == 0


def EpiTrigger(epi):
    return epi % 100 == 0


if __name__ == "__main__":
    train_num = input("input train_number\n")
    USE_Algorithm = input("input your favorite algorithm\n")
    main()

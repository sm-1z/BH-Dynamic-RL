import gymnasium as gym
import stable_baselines3 as sb3
import time
import bh_dyn as bh
from gymnasium.wrappers.record_video import RecordVideo

train_num = "1"

logdir = "./logs/"
videodir = "./videos/BH/"

VIDEO_FLAG = False
USE_DDPG = True


def main():
    if USE_DDPG:
        model = Train("DDPG")
    else:
        model = Train("A2C")
    # env = gym.make("bh_dyn/BhDyn-v0", render_mode="rgb_array")
    # model = sb3.DDPG.load("./BH/DDPG_model", env)
    Test(model)


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
                env,
                videodir + "DDPG_" + train_num,
                episode_trigger=StepTrigger,
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
            tensorboard_log=logdir + "BH",
        )
        # total_timesteps=1e6
        model.learn(total_timesteps=50000)

        model.save("./BH/DDPG_model" + train_num)

    elif method == "A2C":
        if VIDEO_FLAG:
            env = RecordVideo(
                env, videodir + "A2C" + train_num, step_trigger=StepTrigger
            )
        model = sb3.A2C(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=8,
            tensorboard_log=logdir + "BH",
        )
        # total_timesteps=1e6
        model.learn(total_timesteps=50000)

        model.save("./BH/A2C_model" + train_num)

    return model


def Test(model):
    if VIDEO_FLAG:
        env = gym.make(
            "bh_dyn/BhDyn-v0", render_mode="rgb_array", max_episode_steps=3000
        )
        AL = "DDPG" if USE_DDPG else "A2C"
        env = RecordVideo(
            env,
            videodir + AL + "Test" + train_num,
        )  # 不需要env.close()
    else:
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
        time.sleep(0.1)
        print(f"Total Reward: {total_reward}")

    env.close()


def StepTrigger(step):
    return step % 2 == 0


if __name__ == "__main__":
    train_num = input("input train_number\n")
    main()

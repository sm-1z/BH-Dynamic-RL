import gymnasium as gym
import stable_baselines3 as sb3
import time
import bh_dyn as bh
from gymnasium.wrappers.record_video import RecordVideo

train_num = "1"

logdir = "./logs/"
videodir = "./videos/BH/"

VIDEO_FLAG = False
USE_Algorithm = True


def main():

    env = gym.make("bh_dyn/BhDyn-v0", render_mode="human")
    model = sb3.SAC.load("./BH/SAC_model1_5", env)
    Test(model)


def Test(model):
    if VIDEO_FLAG:
        env = gym.make(
            "bh_dyn/BhDyn-v0", render_mode="rgb_array", max_episode_steps=3000
        )
        env = RecordVideo(
            env,
            videodir + "Test" + train_num,
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
    return step % 5000 == 0


def EpiTrigger(epi):
    return epi % 100 == 0


if __name__ == "__main__":
    main()

import mo_gymnasium as mo_gym
import numpy as np
import stable_baselines3 as sb3
import time
from morl_baselines.multi_policy.capql.capql import CAPQL
from stable_baselines3.common import utils, logger
from gymnasium.wrappers.record_video import RecordVideo
import os
import wandb

os.environ["WANDB_API_KEY"] = "1817ddd953f14d7128fde2e15bd0d139458349ea"
os.environ["WANDB_MODE"] ="offline"

GAMMA = 0.99
train_num = "1"
logdir = "./logs/BH/"
videodir = "./videos/BH/"


def main():
    model = Train()
    Test(model)


def Train():
    env = mo_gym.make("mo-bh_dyn-v0", render_mode="rgb_array")
    env = RecordVideo(
        env, videodir + "CAPQL_" + train_num, episode_trigger=Trigger
    )
    model = CAPQL(
                env=env,
                gamma=GAMMA,
                log=False,  # use weights and biases to see the results!
                learning_starts=10000,
                buffer_size=2000000,
                batch_size=256,
                alpha=0.005,
                wandb_entity="207185099",
                device= "cuda",
                seed=3407,
                train_num=train_num,
                iteration=0,
            )
    if model_name != "":
        model.load(model_name)
        print(f"Load model {model_name} Successfully!")

    
    #wandb.init()
    
    for i in range(40):
        model.iteration = i
        print("Running iteration:", i)
        model.train(
            total_timesteps=50000,
            eval_env=env,
            ref_point=np.array([10, 2, 20, 5]),
            # num_eval_episodes_for_front=5,
            checkpoints=True,
        )
        model.save("./model/BH" + train_num, "CAPQL"+train_num)
        print("Finish running iteration:", i, "save model")
    # model.load("./weights/CAPQL.tar")
    return model


def Test(model):
    env = mo_gym.make("mo-humanoid-v4", render_mode="human")
    env = RecordVideo(
        env,
        videodir + "CAPQL_" + "test",
    )

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

    # env.close()


def Trigger(step):
    return step % 100 == 0


if __name__ == "__main__":
    train_num = input("input train_number\n")
    model_name = input("type your model name:\n")
    main()


# def main():

#     env = mo_gym.make("mo-halfcheetah-v4", render_mode="human")
#     model = CAPQL(
#         env=env,
#         gamma=GAMMA,
#         log=False,  # use weights and biases to see the results!
#     )
#     model.train(total_timesteps=500, eval_env=env, ref_point=np.array([0, 50]))
#     model.save("./mo-halfcheetah/CAPQL_model/")

#     time.sleep(5)
#     print("Again")
#     model.load("./mo-halfcheetah/CAPQL_model/CAPQL.tar")
#     # env.reset()

#     model.train(total_timesteps=500, eval_env=env, ref_point=np.array([0, 50]))


# if __name__ == "__main__":
#     main()

# Sample test

# env = mo_gym.MORecordEpisodeStatistics(env) # 一轮循环结束后输出特定信息

# obs, _ = env.reset()
# done = False
# total_reward = 0

# while not done:
#     action = env.action_space.sample()
#     obs, reward, done1, done2, info = env.step(action)
#     total_reward += reward
#     done = done1 or done2
#     print("reward: ", reward)
#     # print(f"Total Reward: {total_reward}")

# env.close()


# sb3 test

# def main():
#     model = Train()
#     Test(model)


# def Train():
#     env = mo_gym.make("mo-halfcheetah-v4", render_mode="human")
#     env = mo_gym.LinearReward(env, np.array([1, 1]))  # 奖励值线性化为标量

#     model = sb3.A2C("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=5000)
#     model.save("./mo-halfcheetah/A2C_model")
#     return model


# def Test(model):
#     env = mo_gym.make("mo-halfcheetah-v4", render_mode="human")
#     env = mo_gym.LinearReward(env, np.array([1, 1]))  # 奖励值线性化为标量

#     obs, _ = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _obs = model.predict(obs, deterministic=True)
#         obs, reward, done1, done2, info = env.step(action)
#         total_reward += reward
#         done = done1 or done2
#         print(f"Total Reward: {total_reward}")
#         env.close()


# if __name__ == "__main__":
#     main()

import gymnasium as gym
import stable_baselines3 as sb3
import time
from gymnasium.wrappers.record_video import RecordVideo

train_num = "1"
logdir = "./logs/Humanoid/"
videodir = "./videos/Humanoid/"

VIDEO_FLAG = False
USE_DDPG = True


def main():
    if USE_DDPG:
        model = Train("DDPG")
    else:
        model = Train("A2C")
    # env = gym.make("Humanoid-v4", render_mode="rgb_array")
    # model = sb3.DDPG.load("./Humanoid/DDPG_model", env)
    Test(model)


def Train(method):
    if VIDEO_FLAG:
        env = gym.make(
            "Humanoid-v4", render_mode="rgb_array", max_episode_steps=2000
        )
    else:
        env = gym.make(
            "Humanoid-v4", render_mode="human", max_episode_steps=2000
        )

    if method == "DDPG":
        if VIDEO_FLAG:
            env = RecordVideo(
                env,
                videodir + "DDPG_" + train_num,
                step_trigger=StepTrigger,
            )
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
            tensorboard_log=logdir,
        )
        # total_timesteps=1e6
        model.learn(total_timesteps=1000000)

        model.save("./Humanoid/DDPG_model" + train_num)

    elif method == "A2C":
        if VIDEO_FLAG:
            env = RecordVideo(
                env,
                videodir + "A2C_" + train_num,
                step_trigger=StepTrigger,
            )
        model = sb3.A2C(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=8,
            tensorboard_log=logdir,
        )
        # total_timesteps=1e6
        model.learn(total_timesteps=5000)

        model.save("./Humanoid/A2C_model" + train_num)

    return model


def Test(model):

    env = env = gym.make("Humanoid-v4", render_mode="rgb_array")
    env = RecordVideo(
        env,
        videodir + "DDPG" + "Test" + train_num,
    )  # 不需要env.close()

    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:

        action, _obs = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward
        done = done1 or done2
        time.sleep(0.5)
        print(f"Total Reward: {total_reward}")

    # env.close()


def StepTrigger(step):
    return step % 1000 == 0


if __name__ == "__main__":
    train_num = input("input train_number\n")
    main()


# simple test

# env = gym.make("Humanoid-v4", render_mode="human")

# obs, _ = env.reset()
# done = False

# for epi in range(100):

#     obs, _ = env.reset()
#     done = False

#     while not done:
#         env.render()

#         action = env.action_space.sample()
#         obs, reward, x, y, _ = env.step(action)
#         print(reward)
#         done = x or y

# env.close()

# test

# env = gym.make("Humanoid-v4", render_mode="rgb_array")

# model = sb3.DDPG("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=5000)

# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")

# vec_env.close()

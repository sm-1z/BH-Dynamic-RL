import gymnasium as gym
import stable_baselines3 as sb3
from gymnasium.wrappers.record_video import RecordVideo

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
import numpy as np

import envs

import os

model_choose = "Cassie-bh-new-v2"

logdir = "./logs/Cassie/"
videodir = "./videos/Cassie/"
modeldir = "./models/Cassie/"

train_num = ""
USE_Algorithm = None
flag = ""

VIDEO_FLAG = True
STEP_TRI_NUM = 5000
EPI_TRI_NUM = 5000
MAX_EPI_STEP = 3000

class SaveInfoCallback(BaseCallback):
    def __init__(self, check_freq: int, save_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, "plot_data.npz")
        self.CoT_values = []
        self.contact_ext_force_values = []
        self.control_torque_values = []
        self.stability_values = []
        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []
        self.epi_len = []
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        info = self.locals.get('infos', None)
        if self.n_calls % self.check_freq == 0:
            if info is not None:
                # Assuming you want to log a specific info key, e.g., "time"
                CoT = [inf.get('CoT', np.nan) for inf in info]
                self.CoT_values.extend(CoT)
                # for value in CoT:
                #     print(f"COT={value:.2f}")
                contact_ext_force = [inf.get('contact_ext_force', np.nan) for inf in info]
                self.contact_ext_force_values.append(contact_ext_force)
                control_torque = [inf.get('control_torque', np.nan) for inf in info]
                self.control_torque_values.append(control_torque)
                stability = [inf.get('stability', np.nan) for inf in info]

            self.timesteps.append(self.model.num_timesteps)
            self.stability_values.append(stability)
            mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
            _, epi_len = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10, return_episode_rewards=True)
            # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
            self.mean_rewards.append(mean_reward)
            self.std_rewards.append(std_reward)
            self.epi_len.append(epi_len)

            np.savez(self.save_path, timesteps=self.timesteps, mean_rewards=self.mean_rewards, std_rewards=self.std_rewards,
                     epi_len=self.epi_len, CoT=self.CoT_values, contact_ext_force=self.contact_ext_force_values,
                     control_torque=self.control_torque_values, stability=self.stability_values)
            
        return True
    



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
            model_choose,
            render_mode="rgb_array",
            max_episode_steps=MAX_EPI_STEP,
        )
        eval_env = gym.make(
            model_choose,
            render_mode="rgb_array",
            max_episode_steps=MAX_EPI_STEP,
        )
    else:
        env = gym.make(
            model_choose, render_mode="human", max_episode_steps=MAX_EPI_STEP
        )
        eval_env = gym.make(
            model_choose,
            render_mode="human",
            max_episode_steps=MAX_EPI_STEP,
        )
    save_info_callback = SaveInfoCallback(check_freq=2000, save_dir=os.path.join(modeldir, USE_Algorithm, train_num))

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=os.path.join(modeldir, USE_Algorithm, train_num),
                                             save_replay_buffer=False,save_vecnormalize=False)
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(modeldir, USE_Algorithm, train_num), 
                                 log_path=os.path.join(logdir, USE_Algorithm, train_num), eval_freq=500, n_eval_episodes=10, deterministic=True, render=False)
    
    callbacklist = CallbackList([save_info_callback, checkpoint_callback, eval_callback])
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
            action_noise=NormalActionNoise(
                np.zeros((17,)), 0.1 * np.ones((17,)) # 需要改
            ),
            train_freq=1,
            gradient_steps=1,
            learning_rate=1e-3,
            batch_size=256,
            policy_kwargs=dict(net_arch=[400, 300]),
            verbose=1,
            tensorboard_log=os.path.join(logdir, "TD3", train_num),
        )

        model.learn(total_timesteps=2e6, progress_bar=True, callback=callbacklist)
        path = os.path.join(modeldir, "TD3", train_num)
        model.save(path)
        del model
        # for i in range(40):
        #     model.learn(total_timesteps=50000)
        #     path = os.path.join(modeldir, "TD3", train_num, str(i))
        #     # 保存模型
        #     model.save(path)

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
        
        model.learn(total_timesteps=2e6, progress_bar=True, callback=callbacklist)
        path = os.path.join(modeldir, "A2C", train_num)
        model.save(path)
        del model
        # for i in range(40):
        #     model.learn(total_timesteps=50000)
        #     path = os.path.join(modeldir, "A2C", train_num, str(i))
        #     model.save(path)

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
        model.learn(total_timesteps=2e6, progress_bar=True, callback=callbacklist)
        path = os.path.join(modeldir, "SAC", train_num)
        model.save(path)
        del model
        # for i in range(40):
        #     model.learn(total_timesteps=5000)
        #     path = os.path.join(modeldir, "SAC", train_num, str(i))
        #     model.save(path)

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

def Test():
    env= gym.make(model_choose, render_mode="rgb_array", max_episode_steps=MAX_EPI_STEP)
    
    if USE_Algorithm == "TD3":
        model = sb3.TD3.load(modeldir + Test_Model, env=env)
    elif USE_Algorithm == "A2C":
        model = sb3.A2C.load(modeldir + Test_Model, env=env)
    elif USE_Algorithm == "SAC":
        model = sb3.SAC.load(modeldir + Test_Model, env=env)
    else:
        raise Exception("Not have this algorithm！")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    vec_env = model.get_env()
    obs = vec_env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    CheckDir()
    flag = input("train or test?\n")
    if flag == 'train':
        train_num = input("input train_number\n")
        USE_Algorithm = input("input your favorite algorithm\n")
        main()
    elif flag == 'test':
        USE_Algorithm = input("input your favorite algorithm\n")
        Test_Model = input("Test model\n")
        Test()
    else:
        raise Exception("Not have this input！")

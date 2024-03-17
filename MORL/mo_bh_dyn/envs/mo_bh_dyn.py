import numpy as np
import mo_bh_dyn.envs.BH_model as bh
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOBHDynEnv(bh.BhModelEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HalfCheetahEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) for more information.

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward
    - 1: Control cost of the action
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.reward_dim = 3

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        vec_reward = np.array(
            [
                info["forward_reward"],
                info["robustness_reward"],
                info["effic_reward"],
            ],
            dtype=np.float32,
        )
        return observation, vec_reward, terminated, truncated, info

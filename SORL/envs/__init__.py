from .humanoid_bh import HumanoidEnv
from gymnasium.envs.registration import register

register(
    id="Humanoid-bh",
    entry_point="envs:HumanoidEnv",
)

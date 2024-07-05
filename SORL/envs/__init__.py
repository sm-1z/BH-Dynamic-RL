from .humanoid_bh import HumanoidEnv
from .cassie_bh import CassieEnv
from .cassie_bh_v2 import CassieEnv2
from gymnasium.envs.registration import register

register(
    id="Humanoid-bh",
    entry_point="envs:HumanoidEnv",
)

register(
    id="Cassie-bh",
    entry_point="envs:CassieEnv",
)

register(
    id="Cassie-bh-new-v2",
    entry_point="envs:CassieEnv2",
)
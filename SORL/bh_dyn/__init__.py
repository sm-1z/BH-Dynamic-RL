from gymnasium.envs.registration import register

register(
    id="bh_dyn/BhDyn-v0",
    entry_point="bh_dyn.envs:BhModelEnv",
)

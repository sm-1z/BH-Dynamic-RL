from gymnasium.envs.registration import register


register(
    id="mo-bh_dyn-v0",
    entry_point="mo_bh_dyn.envs.mo_bh_dyn:MOBHDynEnv",
    max_episode_steps=3000,
)

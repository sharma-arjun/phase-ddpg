from gym.envs.registration import register

register(
    id='Hopper-v2',
    entry_point='phase_envs.envs:HopperEnv2',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Ant-v2',
    entry_point='phase_envs.envs:AntEnv2',
    max_episode_steps=400,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v2',
    entry_point='phase_envs.envs:HumanoidEnv',
    max_episode_steps=400,
)

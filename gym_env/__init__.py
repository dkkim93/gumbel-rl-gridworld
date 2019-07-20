from gym.envs.registration import register


########################################################################################
# GRIDWORLD
register(
    id='Gridworld-v0',
    entry_point='gym_env.gridworld.gridworld_env:GridworldEnv',
    kwargs={'row': 1, 'col': 1},
    max_episode_steps=10
)

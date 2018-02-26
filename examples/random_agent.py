import gym
import gym_rubik.envs.rubik_env

import matplotlib.pyplot as plt

env = gym.make('RubikEnv-v0')

# Configuration
solved = 0
num_episodes = 10000
print_every = 20
timestep_limit = 25
scramble_size = 5
render_cube = False
debug_level = gym_rubik.envs.DebugLevel.WARNING

# Saved stats
x = []
y = []

env.config(debug_level=debug_level, render_cube=render_cube, scramble_size=scramble_size)

for i_episode in range(num_episodes):
    # print("Episode " + str(i_episode) + ". resetting.")
    observation = env.reset()
    # print(observation)
    actions = []
    for t in range(timestep_limit):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        actions.append(env.unwrapped.action_name(action))
        observation, reward, done, info = env.step(action)
        if done:
            solved += 1
            print("Episode solved after {0} timesteps ; solved: {1}/{2}".format(t + 1, str(solved), str(i_episode + 1)))
            if debug_level == gym_rubik.envs.DebugLevel.INFO:
                print("actions:" + ', '.join(actions))
                print("scramble:" + ', '.join(env.unwrapped.get_scramble()))

            x.append(i_episode + 1)
            y.append(solved)
            break

    if (i_episode + 1) % print_every == 0:
        print("Episode not solved after {0} timesteps ; solved: {1}/{2}".format(timestep_limit, str(solved),
                                                                                str(i_episode + 1)))
        x.append(i_episode + 1)
        y.append(solved)

plt.plot(x, y)
plt.show()

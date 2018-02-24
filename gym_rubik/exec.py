import gym
import gym_rubik.envs.rubik_env

env = gym.make('RubikEnv-v0')

solved = 0
num_episodes = 10000000
print_every = 1000
num_moves = 38
scramble_size = 10
render_cube = False

env.config(debug_level=gym_rubik.envs.DebugLevel.WARNING, render_cube=render_cube, scramble_size=scramble_size)

for i_episode in range(num_episodes):
    # print("Episode " + str(i_episode) + ". resetting.")
    observation = env.reset()
    for t in range(num_moves):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            solved += 1
            print("Episode solved after {0} timesteps ; solved: {1}/{2}".format(t + 1, str(solved), str(i_episode + 1)))
            break
        if (i_episode + 1) % print_every == 0 and t == num_moves - 1:
            print("Episode not solved after {0} timesteps ; solved: {1}/{2}".format(t + 1, str(solved),
                                                                                    str(i_episode + 1)))

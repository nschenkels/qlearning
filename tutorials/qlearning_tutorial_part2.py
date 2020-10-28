"""
qlearning_tutorial_part2.py

The second part of the tutorial:
https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/
"""

import gym
import numpy as np

env = gym.make("MountainCar-v0")

# Learning parameters:
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

# Visualization:
SHOW_EVERY = 2000

#
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE

# Randomness:
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING -
                               START_EPSILON_DECAYING)

# Initialize Q-table:
q_table = np.random.uniform(low=-2, high=0,
                            size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (
        state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    # Visualization:
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    # Reset environment:
    discrete_state = get_discrete_state(env.reset())
    done = False

    #
    while not done:
        # Decide on action:
        if np.random.random() > epsilon:
            # Action from Q-table:
            action = np.argmax(q_table[discrete_state])
        else:
            # Random action:
            action = np.random.randint(0, env.action_space.n)

        # Perform action:
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        # Visualization:
        if render:
            env.render()

        # Update Q-table if not finished:
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE)*current_q + \
                LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f'We made it on episode {episode}.')
            q_table[discrete_state + (action,)] = 0

        # Update current state:
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()

"""
qlearning_tutorial_part3.py

The third part of the tutorial:
https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
"""

import os
import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

sns.set()

def learning(save_q_tables=False):
    #
    env = gym.make("MountainCar-v0")

    # Learning parameters:
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EPISODES = 200000

    # Visualization:
    SHOW_EVERY = 1000

    #
    DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high -
                             env.observation_space.low)/DISCRETE_OS_SIZE

    # Randomness:
    epsilon = 0.75
    START_EPSILON_DECAYING = EPISODES//10
    END_EPSILON_DECAYING = EPISODES//3
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING -
                                   START_EPSILON_DECAYING)

    # Initialize Q-table:
    q_table = np.random.uniform(low=-2, high=0,
                                size=(DISCRETE_OS_SIZE + [env.action_space.n]))

    # Monitoring:
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

    def get_discrete_state(state):
        discrete_state = (
            state - env.observation_space.low)/discrete_os_win_size
        return tuple(discrete_state.astype(np.int))

    for episode in range(EPISODES):
        # Monitoring:
        episode_reward = 0

        # Visualization:
        if episode % SHOW_EVERY == 0:
            # print(episode)
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
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)

            # Visualization:
            # if render:
            #     env.render()

            # Update Q-table if not finished:
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - LEARNING_RATE)*current_q + \
                    LEARNING_RATE*(reward + DISCOUNT*max_future_q)
                q_table[discrete_state + (action,)] = new_q
            elif new_state[0] >= env.goal_position:
                # print(f'We made it on episode {episode}.')
                q_table[discrete_state + (action,)] = 0

            # Update current state:
            discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        ep_rewards.append(episode_reward)

        if not episode % SHOW_EVERY or episode == EPISODES - 1:
            if save_q_tables:
                np.save(f'qtables/qtable_{episode}.npy', q_table)
            average_reward = sum(ep_rewards[-SHOW_EVERY:])/ \
                len(ep_rewards[-SHOW_EVERY:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

            print(f"Episode: {aggr_ep_rewards['ep'][-1]} avg: "
                  f"{aggr_ep_rewards['avg'][-1]} min: "
                  f"{aggr_ep_rewards['min'][-1]} max: "
                  f"{aggr_ep_rewards['max'][-1]}")

    env.close()

    plt.figure()
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], '-x', label='avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], '-x', label='min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], '-x', label='max')
    plt.legend(loc=4)
    plt.show()
    if save_q_tables:
        plt.savefig('qlearning_figure_part3.png')



def make_figures():
    def get_q_color(value, vals):
        if value == max(vals):
            return "green", 1.0
        else:
            return "red", 0.3

    fig = plt.figure(figsize=(12, 9))

    qtables = []
    for filename in os.listdir('qtables/'):
        index = int(filename.split('_')[1][:-4])
        qtables.append((index, filename))
    qtables = sorted(qtables, key=lambda v: v[0])

    for index, filename in qtables:
        index = filename.split('_')[1][:-4]
        print(index, filename)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        q_table = np.load(f'qtables/{filename}')

        for x, x_vals in enumerate(q_table):
            for y, y_vals in enumerate(x_vals):
                ax1.scatter(x, y, marker="o",
                            c=get_q_color(y_vals[0], y_vals)[0],
                            alpha=get_q_color(y_vals[0], y_vals)[1])
                ax2.scatter(x, y, marker="o",
                            c=get_q_color(y_vals[1], y_vals)[0],
                            alpha=get_q_color(y_vals[1], y_vals)[1])
                ax3.scatter(x, y, marker="o",
                            c=get_q_color(y_vals[2], y_vals)[0],
                            alpha=get_q_color(y_vals[2], y_vals)[1])

                ax1.set_ylabel("Action 0")
                ax2.set_ylabel("Action 1")
                ax3.set_ylabel("Action 2")

        #plt.show()
        plt.savefig(f'charts/chart_{index}.png')
        plt.clf()

def make_video():
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('video_part3.avi', fourcc, 10.0, (1200, 900))

    charts = sorted(
        os.listdir('charts/'),
        key=lambda filename: int(filename.split('_')[1][:-4]))

    for filename in charts:
        img_path = f"charts/{filename}"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


if __name__ == '__main__':
    learning(save_q_tables=False)
    # make_figures()
    # make_video()

"""
agent2048.py
"""


import os
import cv2
import random
import argparse
import datetime
import numpy as np
import seaborn as sns
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt


from game2048 import Game2048


sns.set()


# Agent class
class Agent2048:
    """
    Agent class
    """
    def __init__(self, env, name=None, epsilon_callback=None, **kwargs):
        """
        epsilon_callback: a function that takes the current value of epsilon
                          and the current episode as input and returns a new
                          value for epsilon.
        **kwargs: can be used to override the default training parameters.
        """

        # Training parameters:
        self.EPISODES = 10_000
        self.EPSILON_DECAY = 0.999
        self.EPSILON_MIN = 0.001
        self.GAMMA = 0.99
        self.MAXIMUM_MOVES_PER_GAME = 500
        self.MEMORY_SIZE_MAX = 1_000_000
        self.MEMORY_SIZE_MIN = 1_000
        self.MINIBATCH_SIZE = 128
        self.RENDER = True
        self.RENDER_EVERY = 100
        self.RENDER_SAVE = False
        self.STATS_AGGREGATION_PERIOD = 100
        self.SAVE = True
        self.SAVE_EVERY = 100
        self.TARGET_MODEL_UPDATE_PERIOD = 42

        # Override default training parameters:
        for kwarg in kwargs:
            if hasattr(self, kwarg):
                setattr(self, kwarg, kwargs[kwarg])

        # Process input:
        name = name.replace(' ', '_') if name is not None else 'Agent2048'
        self.name = \
            name + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
        self.epsilon_callback=epsilon_callback

        #
        self.env = env
        self.epsilon = 1
        self.memory = deque(maxlen=self.MEMORY_SIZE_MAX)
        self.rewards_per_episode = []
        self.rewards_aggregated_epi = []
        self.rewards_aggregated_avg = []
        self.rewards_aggregated_max = []
        self.rewards_aggregated_min = []
        self.target_model_update_counter = 0

        # Main & target model
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        """
        Create & compile a model.
        """

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=self.env.state.shape,
                                  activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, input_shape=self.env.state.shape,
                                  activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, input_shape=self.env.state.shape,
                                  activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, input_shape=self.env.state.shape,
                                  activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, input_shape=self.env.state.shape,
                                  activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(env.actions), activation='linear')
        ])
        model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
        return model

    def train(self):
        """
        Train the agent to play the game.
        """

        # Iterate over episodes
        for episode in range(1, self.EPISODES + 1):

            # Update episode parameters:
            episode_reward = 0
            episode_step = 1

            # Reset the game environment:
            current_state = self.env.reset()
            done = False

            #
            counter1 = 0
            counter2 = 0
            for move in range(1, self.MAXIMUM_MOVES_PER_GAME + 1):
                # Determine which action to take:
                if np.random.rand() > self.epsilon:
                    action = np.argmax(
                        self.model.predict(np.array([current_state]))[0])
                else:
                    counter1 += 1
                    action = np.random.randint(0, len(self.env.actions))

                # Perform the action:
                new_state, reward, done, info = self.env.step(
                    self.env.actions[action])
                episode_reward += reward
                if reward < 0:
                    counter2 += 1

                # Render the environment:
                if self.RENDER and not episode % self.RENDER_EVERY:
                    self.env.render()
                    if self.RENDER_SAVE:
                        self.env.fig.savefig(f'replays/{move}.png')

                # Update the memory and train the networks:
                self.memory.append(
                    (current_state, action, reward, new_state, done, info))
                self.__train_networks__(done, episode_step)

                #
                current_state = new_state
                episode_step += 1

                #
                if done:
                    break

            # Save a video replay of the episode:
            if self.RENDER and not episode % self.RENDER_EVERY and \
                    self.RENDER_SAVE:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter(
                    f'replays/replay_{self.name}_episode_{episode:04}.avi',
                    fourcc, 5, (400, 400))
                frames = [frame for frame in os.listdir('replays/') if
                          frame.endswith('.png')]
                for frame in sorted(frames, key=lambda frame: int(frame[:-4])):
                    out.write(cv2.imread('replays/' + frame))
                out.release()
                os.system('rm replays/*.png')

            #
            self.rewards_per_episode.append(episode_reward)
            window = self.rewards_per_episode[-self.STATS_AGGREGATION_PERIOD:]
            if not episode % self.STATS_AGGREGATION_PERIOD or \
                    episode == self.EPISODES:
                self.rewards_aggregated_epi.append(episode)
                self.rewards_aggregated_avg.append(np.mean(window))
                self.rewards_aggregated_max.append(np.max(window))
                self.rewards_aggregated_min.append(np.min(window))

            print(f'episode: {episode:04}, moves (t/r/n): {move:03}'
                  f'/{counter1:03}/{counter2:03}, '
                  f'reward: {episode_reward:3}, '
                  f'highest tile: {2**np.max(current_state):4}, '
                  f'min: {np.min(window):3} '
                  f'avg: {np.mean(window):3.2f}, '
                  f'max: {np.max(window):3}, '
                  f'epsilon: {self.epsilon:.3f}')

            # Update epsilon
            if self.epsilon_callback is not None:
                self.epsilon = self.epsilon_callback(self.epsilon, episode)
            elif self.epsilon > self.EPSILON_MIN:
                self.epsilon = max(self.EPSILON_MIN,
                                   self.EPSILON_DECAY*self.epsilon)

            #
            if self.SAVE and (not episode % self.SAVE_EVERY or
                              episode == self.EPISODES) and episode != 1:
                self.save(episode)


    def __train_networks__(self, terminal_state, step):
        """
        Train the main and the target network.
        """

        # Start training only if the memory has reached a certain size:
        if len(self.memory) < self.MEMORY_SIZE_MIN:
            return None

        # Take a minibacht from the memory:
        minibatch = random.sample(self.memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch and the correponding Q values from
        # the model:
        current_states = np.array([memory[0] for memory in minibatch])
        current_qs = self.model.predict(current_states)

        # Get future states from minibatch and the corresponding Q values from
        # the target model:
        future_states = np.array([memory[3] for memory in minibatch])
        future_qs = self.target_model.predict(future_states)

        # Process the minibatch:
        x = []
        y = []
        for index, memory in enumerate(minibatch):
            current_state, action, reward, future_state, done, info = memory

            # Calculate new Q-value:
            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + self.GAMMA*max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_q = current_qs[index]
            current_q[action] = new_q

            # And append to our training data
            x.append(current_state)
            y.append(current_q)

        # Fit the minibatch, log only on terminal state
        self.model.fit(np.array(x), np.array(y), batch_size=self.MINIBATCH_SIZE,
                       verbose=0, shuffle=False)

        # Update target network:
        if terminal_state:
            self.target_model_update_counter += 1
        if self.target_model_update_counter == self.TARGET_MODEL_UPDATE_PERIOD:
            self.target_model.set_weights(self.model.get_weights())
            self.target_model_update_counter = 0

    def save(self, episode):
        self.model.save(f'models/{self.name}_episode_{episode:04d}.model')

    def load(self, model_folder_name):
        self.load(model_folder_name)


def reward_callback(current_state, new_state):
    """
    Custom reward functino for Game2048.
    """

    n1 = len(np.where(current_state == 0)[0])
    n2 = len(np.where(new_state == 0)[0])

    if np.max(new_state) > np.max(current_state):
        # New highest tile:
        return 1
    elif n1 <= n2:
        # Tiles were merged:
        reward = 1
    else:
        reward = -1000

    return reward


def epsilon_callback_1(epsilon, episode):
    """
    Custom epsilon strategy for Agent2048.
    """

    DECAY_PERIOD = 50
    PERIOD = 10*DECAY_PERIOD
    CNST = np.pi/(2*DECAY_PERIOD)

    x = (episode - 1) % PERIOD    # Map episode to {0, 1, ..., PERIOD - 1}
    y = x // DECAY_PERIOD         # Map x to {0, 1, ..., PERIOD/DECAY_PERIOD - 1}
    z = (episode - 1) % DECAY_PERIOD

    min_eps = 1 - (y + 1)*0.1
    factor = 1 - min_eps

    return min_eps + factor*np.cos(CNST*z)

def epsilon_callback_2(epsilon, episode):
    """
    Custom epsilon strategy for Agent2048.
    """

    if episode <= 50:
        return 0.90
    elif episode <= 100:
        return 0.80
    elif episode <= 150:
        return 0.70
    elif episode <= 200:
        return 0.60
    elif episode <= 250:
        return 0.50
    elif episode <= 350:
        return 0.40
    elif episode <= 400:
        return 0.30
    elif episode <= 450:
        return 0.20
    elif episode <= 500:
        return 0.10
    else:
        return 0.05


if __name__ == '__main__':
    # Custom training parameters:
    PARAMS = {
        'EPISODES': 1_000,
        'MAXIMUM_MOVES_PER_GAME': 500,
        'MEMORY_SIZE_MAX': 1_000_000,
        'MEMORY_SIZE_MIN': 10_000,
        'MINIBATCH_SIZE': 512,
        'RENDER_EVERY': 25,
        'RENDER_SAVE': True,
        'STATS_AGGREGATION_PERIOD': 10,
        'SAVE_EVERY': 100,
        'TARGET_MODEL_UPDATE_PERIOD': 10,
    }

    # Setup game environment and agent:
    env = Game2048(reward_callback)
    agent = Agent2048(env, 'test_model', epsilon_callback_2, **PARAMS)

    # Train agent:
    agent.train()
    plt.close('all')
    plt.figure()
    plt.plot(agent.rewards_aggregated_epi, agent.rewards_aggregated_avg, 'C0-x')
    plt.plot(agent.rewards_aggregated_epi, agent.rewards_aggregated_max, 'C1--')
    plt.plot(agent.rewards_aggregated_epi, agent.rewards_aggregated_min, 'C1--')
    plt.title(f'Aggregated reward (period = {agent.STATS_AGGREGATION_PERIOD})')
    plt.xlabel('Episodes')
    plt.ylabel('Aggregated reward')
    plt.savefig(f'models/summary_{agent.name}.png')

    # Load agent: (TO DO)
    # agent.load()

    # Have the agent play a game: (TO DO)
    # agent.play()

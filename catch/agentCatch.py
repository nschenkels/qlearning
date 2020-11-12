"""
catch_agent.py
"""


import os
import cv2
import pickle
import random
import argparse
import datetime
import numpy as np
import seaborn as sns
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt


from gameCatch import Catch


sns.set()


class CatchAgent:
    """
    Agent class
    """
    def __init__(self, env, name=None, **kwargs):
        """
        **kwargs: can be used to override the default parameters.
        """

        # Training parameters:
        self.EPISODES = 1_000
        self.EPSILON_START = 1
        self.EPSILON_DECAY = 0.999
        self.EPSILON_MIN = 0.001
        self.GAMMA = 0.99
        self.MEMORY_SIZE_MAX = 1_000_000
        self.MEMORY_SIZE_MIN = 1_000
        self.MINIBATCH_SIZE = 128
        self.TARGET_MODEL_UPDATE_PERIOD = 20

        # Rendering parameters:
        self.RENDER = False                   # render the game
        self.RENDER_EVERY = 100
        self.RENDER_SAVE = True
        self.STATS_AGGREGATION_PERIOD = 10
        self.RENDER_STATS = True              # render statistics
        self.RENDER_STATS_SAVE = True

        # Save parameters:
        self.SAVE = True
        self.SAVE_EVERY = 100

       # Output folders:
        self.FOLDER_MODELS = 'models/'
        self.FOLDER_REPLAYS = 'replays/'

        # Process input:
        if name is None:
            self.name = 'catch_agent_' + \
                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        else:
            self.name = name.replace(' ', '_')
        for kwarg in kwargs:
            if hasattr(self, kwarg):
                setattr(self, kwarg, kwargs[kwarg])

        # Make sure output folders exist:
        if not os.path.isdir(self.FOLDER_MODELS):
            os.makedirs(self.FOLDER_MODELS)
        if not os.path.isdir(self.FOLDER_REPLAYS):
            os.makedirs(self.FOLDER_REPLAYS)

        #
        self.env = env
        self.epsilons = []
        self.memory = deque(maxlen=self.MEMORY_SIZE_MAX)
        self.stats = {'rewards': [], 'aggr_avg': [], 'aggr_max': [],
                      'aggr_min': []}
        self.target_model_update_counter = 0
        self.offset = 0

        # Figure for aggregated stats:
        self.fig1 = plt.figure()
        self.ax1 = self.fig1.subplots()

        # Figure for epsilon:
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.subplots()

        # Main & target model
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        """
        Create & compile a model.
        """

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                   input_shape=self.env.state.shape,
                                   data_format='channels_first'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.env.actions), activation='linear')
        ])
        model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
        return model

    def train(self):
        """
        Train the agent to play the game.
        """

        # Iterate over episodes
        episode_offset = self.offset
        episode_offset_max = self.offset + self.EPISODES
        for episode in range(1, self.EPISODES + 1):
            #
            episode_offset += 1

            # Reset the game environment:
            current_state = self.env.reset()
            done = False

            # Get epsilon:
            epsilon = self.EPSILON_START if episode == 1 else \
                max(self.EPSILON_MIN, self.EPSILON_DECAY*self.epsilons[-1])
            self.epsilons.append(epsilon)

            # Reset episode variables:
            move = 0
            c1 = 0
            episode_reward = 0
            while not self.env.done:
                #
                move += 1

                # Determine which action to take:
                if np.random.rand() > epsilon:
                    action = np.argmax(
                        self.model.predict(np.array([current_state]))[0])
                else:
                    c1 += 1
                    action = np.random.randint(0, len(self.env.actions))

                # Perform the action:
                new_state, reward, done, info = self.env.step(
                    self.env.actions[action])
                episode_reward += reward

                # Render the environment:
                if self.RENDER and (not episode % self.RENDER_EVERY or
                                    episode == self.EPISODES):
                    self.env.render()
                    if self.RENDER_SAVE:
                        self.env.fig.savefig(f'{self.FOLDER_REPLAYS}{move}.png')

                # Update the memory and train the networks:
                self.memory.append(
                    (current_state, action, reward, new_state, done, info))
                self.__train_networks__(done)

                #
                current_state = new_state
                if done:
                    break

            # Save a video replay of the episode:
            if self.RENDER and (not episode % self.RENDER_EVERY or
                                episode == self.EPISODES) and self.RENDER_SAVE:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter(
                    f'{self.FOLDER_REPLAYS}{self.name}_{episode_offset:04}.avi',
                    fourcc, 10, (375, 500))
                frames = [frame for frame in os.listdir(self.FOLDER_REPLAYS) if
                          frame.endswith('.png')]
                for frame in sorted(frames, key=lambda frame: int(frame[:-4])):
                    filename = f'{self.FOLDER_REPLAYS}/{frame}'
                    out.write(cv2.imread(filename))
                    os.remove(filename)
                out.release()

            # Update stats:
            self.stats['rewards'].append(episode_reward)
            window = self.stats['rewards'][-self.STATS_AGGREGATION_PERIOD:]
            self.stats['aggr_avg'].append(np.mean(window))
            self.stats['aggr_max'].append(np.max(window))
            self.stats['aggr_min'].append(np.min(window))

            #
            if self.RENDER_STATS:
                #
                rr = range(1, self.offset + episode + 1)

                # Aggregated reward:
                self.ax1.clear()
                self.ax1.plot(rr, self.stats['aggr_avg'], 'C0')
                self.ax1.plot(rr, self.stats['aggr_max'], 'C1--')
                self.ax1.plot(rr, self.stats['aggr_min'], 'C1--')
                self.ax1.set_title('Aggregated reward per '
                                  f'{self.STATS_AGGREGATION_PERIOD} episodes')
                self.ax1.set_xlabel('Episode')
                self.ax1.set_ylabel('min/avg/max')
                plt.draw()
                self.fig1.show()
                plt.pause(1e-3)

                # Epsilon:
                self.ax2.clear()
                self.ax2.plot(rr, self.epsilons, 'C0')
                self.ax2.set_title('Exploration level')
                self.ax2.set_xlabel('Episode')
                self.ax2.set_ylabel(r'$\varepsilon$')
                plt.draw()
                self.fig2.show()
                plt.pause(1e-3)

            # Print some information:
            print(f'episode: {episode_offset:4}/{episode_offset_max}, '
                  f'moves (total/random): {move:03}/{c1:03}, '
                  f'score: {self.env.info["score"]:2}, '
                  f'reward: {episode_reward:2}, '
                  f'min: {np.min(window):2} '
                  f'avg: {np.mean(window):2.2f}, '
                  f'max: {np.max(window):2}, '
                  f'epsilon: {epsilon:.3f}')

            # Save model:
            if self.SAVE and (not episode % self.SAVE_EVERY or
                              episode == self.EPISODES) and episode != 1:
                self.save(episode_offset)

        # Save stats figure:
        if self.RENDER_STATS and self.RENDER_STATS_SAVE:
            self.fig1.savefig(self.FOLDER_MODELS + self.name + '_reward.png')
            self.fig2.savefig(self.FOLDER_MODELS + self.name + '_epsilon.png')

    def __train_networks__(self, terminal_state):
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
        model_name = self.FOLDER_MODELS + self.name + f'_episode_{episode:04d}'
        self.model.save_weights(model_name + '.h5')
        data = {'epsilons': self.epsilons,
                'memory': self.memory,
                'stats': self.stats}
        with open(model_name + '.agent', 'wb') as filename:
            pickle.dump(data, filename)

    def load(self, model_name):
        prefix = self.FOLDER_MODELS + model_name
        self.model.load_weights(prefix + '.h5')
        self.target_model.load_weights(prefix + '.h5')
        with open(prefix + '.agent', 'rb') as filename:
            data = pickle.load(filename)
        self.epsilons = data['epsilons']
        self.memory = data['memory']
        self.stats = data['stats']
        self.offset = len(self.epsilons)
        del(data)


if __name__ == '__main__':
    # Custom training parameters:
    PARAMS = {
        'EPISODES': 500,
        'GAMMA': 0.95,
        'EPSILON_START': 0.10,
        'EPSILON_DECAY': 0.999,
        'EPSILON_MIN': 0.01,
        'MEMORY_SIZE_MAX': 1_000_000,
        'MEMORY_SIZE_MIN': 500,
        'MINIBATCH_SIZE': 32,
        'TARGET_MODEL_UPDATE_PERIOD': 10,
        'RENDER': True,
        'RENDER_EVERY': 100,
        'RENDER_SAVE': True,
        'STATS_AGGREGATION_PERIOD': 10,
        'RENDER_STATS': True,
        'RENDER_STATS_SAVE': True,
        'SAVE': True,
        'SAVE_EVERY': 100
    }

    # Setup game environment and agent:
    env = Catch(rounds=10)
    agent = CatchAgent(env, 'model A', **PARAMS)
    agent.load('model_A_episode_0600')

    # Train agent:
    agent.train()

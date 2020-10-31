"""

"""


import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import deque


from game2048 import Game2048


# Agent class
class Agent2048:
    """
    Agent class
    """
    def __init__(self, env, model_name='agent2048', epsilon_callback=None):
        # Training parameters:
        self.EPISODES = 10_000
        self.EPSILON_DECAY = 0.99
        self.EPSILON_MIN = 0.001
        self.GAMMA = 0.99
        self.MAXIMUM_MOVES_PER_GAME = 2000
        self.MEMORY_SIZE_MAX = 100_000
        self.MEMORY_SIZE_MIN = 1_000
        self.MINIBATCH_SIZE = 64
        self.STATS_AGGREGATION_PERIOD = 20
        self.TARGET_MODEL_UPDATE_PERIOD = 42
        self.RENDER = False
        self.RENDER_EVERY = 100

        #
        self.env = env
        self.epsilon = 1
        self.epsilon_callback=epsilon_callback
        self.memory = deque(maxlen=self.MEMORY_SIZE_MAX)
        self.model_name = model_name
        self.rewards_per_episode = []
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
        # for episode in tqdm(range(1, self.EPISODES + 1), ascii=True,
        #                     unit='episodes', ncols=79):
        for episode in range(1, self.EPISODES + 1):

            # Update episode parameters:
            episode_reward = 0
            episode_step = 1

            # Reset the game environment:
            current_state = self.env.reset()
            done = False

            #
            for move in range(1, self.MAXIMUM_MOVES_PER_GAME + 1):
                # Determine which action to take:
                if np.random.rand() > self.epsilon:
                    action = np.argmax(
                        self.model.predict(np.array([current_state]))[0])
                else:
                    action = np.random.randint(0, len(self.env.actions))

                # Perform the action:
                new_state, reward, done, info = self.env.step(
                    self.env.actions[action])
                episode_reward += reward

                # Render the environment:
                if self.RENDER and not (episode - 1) % self.RENDER_EVERY:
                    self.env.render()

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

            #
            self.rewards_per_episode.append(episode_reward)
            window = self.rewards_per_episode[-self.STATS_AGGREGATION_PERIOD:]
            if not (episode - 1) % self.STATS_AGGREGATION_PERIOD:
                self.rewards_aggregated_avg = np.mean(window)
                self.rewards_aggregated_max = np.max(window)
                self.rewards_aggregated_min = np.min(window)
                self.save()

            print(f'episode: {episode:4}, moves: {move:3}, '
                  f'reward: {episode_reward:5}, '
                  f'highest tile: {2**np.max(current_state):4}, '
                  f'min: {np.min(window):2} '
                  f'avg: {np.mean(window):.2f}, '
                  f'max: {np.max(window):2}, '
                  f'epsilon: {self.epsilon:.3f}')

            # Update epsilon
            if self.epsilon_callback is not None:
                self.epsilon = self.epsilon_callback(episode)
            elif self.epsilon > self.EPSILON_MIN:
                self.epsilon = max(self.EPSILON_MIN,
                                   self.EPSILON_DECAY*self.epsilon)

        #
        self.save()


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

    def save(self):
        self.model.save(f'models/{self.model_name}.model')

    def load(self):
        self.model.load_model(f'model/{self.model_name}.model')


def reward_callback(current_state, new_state):
    """
    """

    if 11 in new_state:
        # reach 2048
        return 1000
    elif np.max(new_state) > np.max(current_state):
        # new highest tile
        return 5
    elif np.array_equal(current_state, new_state):
        return -100
    else:
        return 1

def epsilon_callback(episode):
    DECAY_PERIOD = 50
    CNST = 2*np.pi/DECAY_PERIOD

    cnst1 = 1 - ((episode//50)%11 + 1)*0.1
    cnst2 = 1 - cnst1

    return cnst1 + np.cos(CNST*(episode - 1))*cnst2

if __name__ == '__main__':
    #
    model_name = 'test_agent'

    # Setup game environment and agent:
    env = Game2048(reward_callback)
    agent = Agent2048(env, model_name, epsilon_callback)

    # Train agent: (TO DO)
    agent.train()

    # Load agent: (TO DO)
    # agent.load()

    # Have the agent play a game: (TO DO)
    # agent.play()

"""
catch.py

Implementation of the "Catch" class that can be used to simulate playing a game
of catch.
"""


import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt


PADDLE_WIDTH = 3
PADDLE_LEFT_START = 8
WIDTH = 15
HEIGHT = 20

class Catch():
    """
    Catch game environment.

    The game state will consist of a numpy array of shape (HEIGHT, WIDTH). Empty
    spaces are represented by 0, solid spaces by 1.
    """

    def __init__(self, rounds=10):
        self.rounds=rounds
        self.reset()
        self.actions = {0: 'left', 1: 'right', 2: ''}
        self.fig = plt.figure(figsize=(WIDTH/4, HEIGHT/4))
        self.fig.set_tight_layout(True),

    def __str__(self):
        printstring = ''
        for row in self.state:
            for value in row:
                printstring += 'x' if value == 1 else ' '
        return printstring

    def step(self, action):
        """
        Perfrom an action.
        """

        # Move the paddle:
        if action == 'left':
            self.paddle_left = max(0, self.paddle_left - 1)
        elif action == 'right':
            self.paddle_left = min(self.paddle_left + 1, WIDTH - PADDLE_WIDTH)
        elif action == '':
            pass
        else:
            raise Exception('Unknown action.')

        # Drop the ball:
        self.ball_y += 1

        #
        self.reward = 0
        if self.ball_y == HEIGHT - 1:
            #
            if self.paddle_left <= self.ball_x < self.paddle_left + PADDLE_WIDTH:
                self.info['score'] += 1
                self.reward = 1

            #
            self.info['rounds'] += 1
            if self.info['rounds'] == self.rounds:
                self.done = True

            # Reset the ball:
            self.ball_x = np.random.randint(1, WIDTH)
            self.ball_y = 0
        self.info['total_reward'] += self.reward

        # Redraw the state:
        self.draw_state()

        #
        return self.state, self.reward, self.done, self.info

    def reset(self):
        """
        Reset the state of the environment and return an inital observation.
        """

        self.paddle_left = PADDLE_LEFT_START
        self.ball_x = np.random.randint(1, WIDTH)
        self.ball_y = 0
        self.draw_state()
        self.reward = 0
        self.done = False
        self.info = {'rounds': 0, 'score': 0, 'total_reward': 0}
        return self.state

    def draw_state(self):
        """
        Draw the state.
        """
        self.state = np.zeros((1, HEIGHT, WIDTH))
        self.state[0, -1, self.paddle_left:self.paddle_left + PADDLE_WIDTH] = 1
        self.state[0, self.ball_y, self.ball_x] = 1

    def render(self):
        """
        Visualize the agent.
        """

        #
        self.fig.clf()
        ax = self.fig.subplots()
        ax.axis('off')
        ax.imshow(self.state[0], cmap='gray')
        ax.set_title(f"Score: {self.info['score']}/{self.rounds}")
        plt.draw()
        plt.pause(1e-3)

    def play(self):
        """
        Manually play a game of 2048.
        """

        map = {'left': 'left', 'right': 'right', 'q': 'left', 'd': 'right',
               '': ''}

        #
        print('Let\'s play a game of catch!\n'
              'Enter "left/q" or "right/d" to make a move.\n'
              'Enter "quit" to quit.')

        #
        self.reset()
        self.render()
        while not self.done:
            #
            action = 'xxx'
            while action not in map or action == 'quit':
                print('Your move: ', end = '')
                action = input()
                if action == 'quit':
                    return None
                elif action not in map:
                    print('Your move must be "left" or "right. '
                          'Please try again.')
            self.step(map[action])

            #
            self.render()

        #
        if self.info['score'] == self.rounds:
            print('CONGRAGULATIONS, you caught every ball!')
        else:
            print('It seems like you dropped a few balls there. '
                  'Better luck next time.')


if __name__ == '__main__':
    env = Catch()
    env.play()

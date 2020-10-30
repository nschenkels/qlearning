"""
game2048.py

Implementation of the "game2048" class that can be used to simulate playing a
game of 2048.
"""


import numpy as np


class Game2048():
    """
    2048 game environment.

    The game state will consist of a numpy array of shape (16,)  with integer
    entries. This represents the log2 of the tile value. We consider empty tiles
    to have value 1, and they are therefore represented by 0.
    """

    def __init__(self):
        self.actions = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}

    def __str__(self):
        # Get the tile values:
        values = [2**n if n != 0 else '' for n in self.state]

        # Create a string to print:
        printstring = '---------------------\n'
        printstring += '|{:>4}|{:>4}|{:>4}|{:>4}|\n'.format(*values[:4])
        printstring += '---------------------\n'
        printstring += '|{:>4}|{:>4}|{:>4}|{:>4}|\n'.format(*values[4:8])
        printstring += '---------------------\n'
        printstring += '|{:>4}|{:>4}|{:>4}|{:>4}|\n'.format(*values[8:12])
        printstring += '---------------------\n'
        printstring += '|{:>4}|{:>4}|{:>4}|{:>4}|\n'.format(*values[12:])
        printstring += '---------------------'
        return printstring

    def step(self, action):
        """
        Perfrom an action. Note that if two tiles with log2 value n are joined,
        they are replaced by a tile with log2 value n + 1.
        """

        # Perform the action to determine the new state:
        if action in ['left', 'right']:
            # Collect the rows and strip the empty tiles:
            rows = []
            for i in range(4):
                rows.append([n for n in self.state[4*i:4*(i + 1)] if n != 0])

            # Slide the tiles together:
            if action == 'left':
                # For a left swipe, we go from left to right:
                for row in rows:
                    i = 0
                    while i < len(row) - 1:
                        if row[i] == row[i + 1]:
                            row[i] = row[i] + 1
                            row.pop(i + 1)
                        i = i + 1

                # Add empty tiles:
                for i in range(4):
                    rows[i] = rows[i] + (4 - len(rows[i]))*[0]
            else:
                # For a right swipe, we go from right to left:
                for row in rows:
                    i = len(row) - 1
                    while 0 < i:
                        if row[i] == row[i - 1]:
                            row[i] = row[i] + 1
                            row.pop(i - 1)
                        i = i - 1

                # Add empty tiles:
                for i in range(4):
                    rows[i] = (4 - len(rows[i]))*[0] + rows[i]

            # Transform the rows into a new state:
            new_state = np.array([n for row in rows for n in row])
        elif action in ['up', 'down']:
            # Collect the columns and strip the empty tiles:
            cols = []
            for i in range(4):
                cols.append([n for n in self.state[i:13 + i:4] if n != 0])

            # Slide the tiles together:
            if action == 'up':
                # For an up swipe, we iterate from left to right:
                for col in cols:
                    i = 0
                    while i < len(col) - 1:
                        if col[i] == col[i + 1]:
                            col[i] = col[i] + 1
                            col.pop(i + 1)
                        i = i + 1

                # Add empty tiles:
                for i in range(4):
                    cols[i] = cols[i] + (4 - len(cols[i]))*[0]
            else:
                # For a down swipe, we iterate from right to left:
                for col in cols:
                    i = len(col) - 1
                    while 0 < i:
                        if col[i] == col[i - 1]:
                            col[i] = col[i] + 1
                            col.pop(i - 1)
                        i = i - 1

                # Add empty tiles:
                for i in range(4):
                    cols[i] = (4 - len(cols[i]))*[0] + cols[i]

            # Collect the new new_state in one list:
            new_state = [col[i] for i in range(4) for col in cols]
        else:
            raise Exception('Unknown action.')

        # Update the game environment:
        new_max = (max(self.state) != max(new_state))
        state_changed = not np.array_equal(self.state, new_state)
        if state_changed:
            self.state = new_state.copy()
            self.calc_score()
            self.add_random_tile()
            self.check_done()
            self.info['moves'] += 1

        # Calculate the reward:
        if new_max == 11:
            self.reward = 100
        elif new_max:
            self.reward = 10
        elif state_changed:
            self.reward = 0
        else:
            self.reward = -100
        self.info['total_reward'] += self.reward

        #
        return (self.state, self.reward, self.done, self.info)

    def reset(self):
        """
        Resets the state of the environment and returns an inital observation.
        """
        self.state = np.zeros(16, dtype=int)
        self.score = 0
        self.reward = 0
        self.done = False
        self.info = {'fours': 0, 'moves': 0, 'score': 0, 'total_reward': 0}
        self.add_random_tile()
        self.add_random_tile()
        return self.state

    def render(self):
        """
        Used to visualize testing the agent.
        """
        print(f"Score: {self.info['score']}, reward: {self.reward}, "
              f"total reward: {self.info['total_reward']}\n{self}\n")

    def add_random_tile(self):
        """
        Replace a random 0 entry in self.observation with a 1 or a 2.
        This corresponds to adding a new 2 or 4 tile to the board.
        """

        # See which tiles are blank and choose one to fill:
        blanks = [index for index in range(16) if self.state[index] == 0]
        index = np.random.choice(blanks)

        # Add a 4-tile with an alpha probability:
        alpha = 0.025
        if np.random.rand() <= alpha:
            self.state[index] = 2
            self.info['fours'] += 1
        else:
            self.state[index] = 1

    def calc_score(self):
        """
        Update the score of the game.
        A tile of 2**n will contribute (n - 1)*2**n points (if n > 1).
        """
        self.info['score'] = 0
        for n in self.state:
            if 1 < n:
                self.info['score'] += (n - 1)*2**n

        # Add a corrections for the 4-tiles that were randomly added:
        self.score -= 4*self.info['fours']

    def check_done(self):
        """
        Check if we have reached a terminal state.
        """

        if 11 in self.state:
            # There is an 2048 tile ==> we won:
            self.done = True
        elif 0 in self.state:
            # There are still empty spaces => a valid move is possible:
            self.done = False
        else:
            # The grid is full and we didn't win yet.
            # We check for up/down moves:
            for i, j in ((i, j) for i in range(3) for j in range(4)):
                if self.state[4*i + j] == self.state[4*(i + 1) + j]:
                    self.done = False
                    return None          # Quick exit on possible move existing.

            # We check for left/right moves:
            for i, j in ((i, j) for i in range(4) for j in range(3)):
                if self.state[4*i + j] == self.state[4*i + j + 1]:
                    self.done = False
                    return None          # Quick exit on possible move existing.

            # There are no more moves:
            self.done = True

    def play(self):
        """
        Manually play a game of 2048.
        """

        #
        print('Let\'s play a game of 2048\n'
              'Enter "left", "right", "up" or "down" to make a move.\n'
              'Enter "quit" to quit.\n')

        #
        self.reset()
        while not self.done:
            #
            print(f"SCORE: {self.info['score']}\n{self}")

            #
            action = ''
            while action not in ['left', 'right', 'up', 'down', 'quit']:
                print('Your move: ', end = '')
                action = input()
                if action == 'quit':
                    return None
                elif action not in ['left', 'right', 'up', 'down']:
                    print('Your move must be "left", "right", "up" or "down". '
                          'Please try again.')
                print()
            self.step(action)

        #
        print(f"SCORE: {self.info['score']}\n{self}\n")

        #
        if 11 in self.state:
            print('CONGRAGULATIONS')
            print('You have reached 2048!!!')
        else:
            print('GAME OVER')
        print(f"Your score is: {self.info['score']}.")


if __name__ == '__main__':
    env = Game2048()
    env.play()

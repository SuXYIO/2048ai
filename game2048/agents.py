import numpy as np


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False, interact=False, train=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction, train=train)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter), "Direction: {}".format(["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
                if interact:
                    input('Press "Enter" to continue')

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction

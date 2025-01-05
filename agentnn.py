import torch as tch
from game2048.agents import Agent
from network import gamenn

class agentnn(Agent):
    def __init__(self, game, display=None, load_file=True):
        if game.size != 4:
            raise Exception(f'{self.__class__.__name__} only works with game size 4')
        super().__init__(game, display)
        self.game = game
        self.model = gamenn()
        if load_file:
            PATH = './saves/state_dict0'
            self.model = tch.load(PATH)

    def step(self):
        board_tensor = tch.from_numpy(self.game.board.flatten()).float()
        output_tensor = self.model(board_tensor)
        output_list = output_tensor.tolist()
        output_max_ind = output_list.index(max(output_list))
        return int(output_max_ind)

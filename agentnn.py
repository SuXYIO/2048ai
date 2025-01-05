import torch as tch
from typing import Optional
from game2048.agents import Agent

class agentnn(Agent):
    def __init__(self, game, file_path:Optional[str], display=None):
        if game.size != 4:
            raise Exception(f'{self.__class__.__name__} only works with game size 4')
        super().__init__(game, display)
        self.game = game
        if file_path != None:
            self.model = tch.load(file_path)

    def step(self):
        board_tensor = tch.from_numpy(self.game.board.flatten()).float()
        board_tensor = board_tensor.unsqueeze(0)
        output_tensor = self.model(board_tensor)
        output_list = output_tensor.squeeze(0).tolist()
        output_max_ind = output_list.index(max(output_list))
        return int(output_max_ind)

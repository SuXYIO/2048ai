from sys import argv
import torch as tch
from network import *
from game2048.game import Game
from agentnn import agentnn
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import PGPE
from evotorch.logging import PandasLogger, StdOutLogger
import matplotlib.pyplot as plt

class train_agentnn(agentnn):
    def __init__(self, model, game):
        super().__init__(game, file_path=None, display=None)
        self.model = model

def train_run(model, size, score_to_win, AgentClass):
    game = Game(size, score_to_win)
    agent = AgentClass(model, game)
    agent.play(verbose=False)
    return game.score

def train_eval_agent(model, game_size=4, score_to_win=None, test_rounds=4):
    scoresum = 0
    for _ in range(test_rounds):
        score = train_run(model, game_size, score_to_win, AgentClass=train_agentnn)
        scoresum += score
    # average score
    return scoresum / test_rounds

if __name__ == '__main__':
    model = tch.load(argv[1])
    problem = NEProblem(
        objective_sense="max",
        network=model,
        network_eval_func=train_eval_agent,
    )
    searcher = PGPE(
        problem,
        popsize=50,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )
    logger_pandas = PandasLogger(searcher)
    logger_stdout = StdOutLogger(searcher, interval=16)
    TIMES = int(argv[3])
    searcher.run(TIMES)
    logger_pandas.to_dataframe()['mean_eval'].plot()
    plt.show()
    trained_model = problem.parameterize_net(searcher.status["center"])
    PATH = argv[2]
    tch.save(trained_model, PATH)

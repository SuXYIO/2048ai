from sys import argv
from game2048.game import Game
from game2048.displays import Display
from agentnn import agentnn

def demo_run(AgentClass, size=4, score_to_win=None, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True, interact=True)
    return game.score

if __name__ == '__main__':
    demo_run(agentnn, filepath=argv[1])

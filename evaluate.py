from sys import argv
from game2048.game import Game
from game2048.agents import RandomAgent
from agentnn import agentnn

def single_run(AgentClass, size=4, score_to_win=None, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=None, **kwargs)
    agent.play(verbose=False)
    return game.score

def eval_agent(agent, score_to_win=None, test_rounds=4, **kwargs):
    scoresum = 0
    for _ in range(test_rounds):
        score = single_run(agent, score_to_win=score_to_win, **kwargs)
        scoresum += score
    # average score
    return scoresum / test_rounds

if __name__ == '__main__':
    ROUNDS = 256
    AGENT = agentnn
    print('Average score of', ROUNDS, 'for', AGENT, ':', eval_agent(AGENT, test_rounds=ROUNDS, file_path=argv[1]))

from game2048.game import Game
from game2048.displays import Display
from agentnn import agentnn

def single_run(AgentClass, size=4, score_to_win=None, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score

def eval_agent(agent, game_size=4, score_to_win=None, test_rounds=4):
    scoresum = 0
    for _ in range(test_rounds):
        score = single_run(agent, size=game_size, score_to_win=score_to_win)
        scoresum += score
    # average score
    return scoresum / test_rounds

if __name__ == '__main__':
    ROUNDS = 64
    print('Average score of ', ROUNDS, ': ', eval_agent(agentnn, test_rounds=ROUNDS))

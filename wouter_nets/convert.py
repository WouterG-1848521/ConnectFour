import numpy as np
import pandas as pd
from connectfour import Game

def as_board(moves):
    states = np.empty((len(moves), 7*6))
    labels = np.empty((len(moves), 2))

    game = Game()
    for _,i in moves.iterrows():
        idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']
        game.play_move(player, move)
        states[idx,:] = game.board.reshape((-1))
        labels[idx,0] = winner
        labels[idx,1] = player

    return (states, labels)

def read_games(fname):
    X = pd.read_csv(fname, names=["game", "idx", "player", "move", "winner"])
    for _, game in X.groupby('game'):
        yield as_board(game)

# own conversion to : [bestmove, boardstate]
def create_boards_with_moves(moves):
    game = Game()

    set = []
    for _, i in moves.iterrows():
        idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']
        game.play_move(player, move)
        guessDepth = 10
        procents = game.winning(player, n=guessDepth)
        bestMove = (0, procents[0])
        for i in range(1,len(procents)):
            if procents[i] > bestMove[1]:
                bestMove = (i, procents[i])
        # print("board:", game.board)
        # print("has best move:", bestMove)
        total = [bestMove[0]]
        total += game.board.reshape((-1)).tolist()
        set.append(total)
        yield(total)
        # set.append([bestMove[0], game.board.reshape((-1))])
    # return set
def read_games_own(fname):
    X = pd.read_csv(fname, names=["game", "idx", "player", "move", "winner"])

    data = []
    for _, game in X.groupby('game'):
        # data += create_boards_with_moves(game)
        yield create_boards_with_moves(game)

    # return data

# same as previous but with yield, to be used more efficiently
# These functions give the boardstates and their best move one by one.
def create_boards_with_moves(moves):
    game = Game()

    for _, i in moves.iterrows():
        idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']
        game.play_move(player, move)
        guessDepth = 100    # TODO: higher for real training
        procents = game.winning(player, n=guessDepth)
        bestMove = (0, procents[0])
        for i in range(1,len(procents)):
            if procents[i] > bestMove[1]:
                bestMove = (i, procents[i])
        # print("board:", game.board)
        # print("has best move:", bestMove)
        total = [bestMove[0]]
        total += game.board.reshape((-1)).tolist()
        yield total
def convert_games_to_setup(fname):
    X = pd.read_csv(fname, names=["game", "idx", "player", "move", "winner"])

    i = 0
    for _, game in X.groupby('game'):
        yield create_boards_with_moves(game)

# X = np.vstack([np.hstack(game) for game in read_games()])
# np.save('c4.npy', X)

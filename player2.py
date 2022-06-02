import connectfour
from enum import Enum
import numpy as np

# Add names of networks here
class Nets(Enum):
    none = 1
    simpleNetwork = 2


class Player2:
    networkName = Nets.none
    game = None

    def __init__(self, game, networkName = Nets.none):
        self.networkName = networkName

        if (game.__class__ != connectfour.Game):
            print(game.__class__)
            raise ValueError('Game not class Game')
        else:
            self.game = game

    # GET MOVE BASED ON THE CHOSEN NETWORK
    def getMove(self):
        if (self.networkName == Nets.none):
            print("Calculating random move")
            return self.game.random_action(legal_only = True)

        elif (self.networkName == Nets.simpleNetwork):
            print("Calculating move in simpleNetwork")

        else:
            raise ValueError("Something went wrong while identifying chosen network")
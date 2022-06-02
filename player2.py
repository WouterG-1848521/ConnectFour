import connectfour as connectfour
from enum import Enum
from neural_nets.testnet import testNet

# Add names of networks here (DON'T FORGET TO IMPORT THEM)
class Nets(Enum):
    none = 1
    testNet = 2


class Player2:
    network = Nets.none
    net = None
    game = None

    def __init__(self, game, networkName = Nets.none):
        # Make sure Game is correct
        if (game.__class__ != connectfour.Game):
            print(game.__class__)
            raise ValueError('Game is not class Game')
        else:
            self.game = game

        # Save network's name
        if (isinstance(networkName, Nets)):
            self.network = networkName
        else:
            raise ValueError("NetworkName '" + str(networkName) + "' not type from enum 'Nets'")

        # Init Network
        if (not self.network == Nets.none):
            try:
                classname = globals()[self.network.name]
                self.net = classname()

                # Train the network
                self.net.train()
                # Evaluate network
                self.net.evaluate()

            except Exception as e:
                print("Exception caught:", e)
                raise ValueError("Something went wrong while initializing chosen network " + self.network.name)

        

    # GET MOVE BASED ON THE CHOSEN NETWORK
    def getMove(self):
        # If no network is chosen, use random action
        if (self.network == Nets.none):
            print("Calculating random move")
            return self.game.random_action(legal_only = True)

        # If network is chosen, try to use it
        else:
            try:
                predict = self.net.predict(self.game.board)
                return predict
            except Exception as e:
                print("Exception caught:", e)
                raise ValueError("Something went wrong while using chosen network: " + self.network.name)
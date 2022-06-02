# Play the game against the computer or let the computer play against the computer.

from connectfour import Game
from player2 import Player2, Nets
import argparse
import numpy as np

# INIT ARGS
parser = argparse.ArgumentParser(description='Play a game of connect four')
parser.add_argument('--second', metavar='S', nargs='?', default=False, const=True,
                    help='Use this if you want to go second')
parser.add_argument('--aiRandom', metavar='A', nargs='?', default=False, const=True,
                    help='Use this if you want to have 2 random ais playing against each other')
args = parser.parse_args()

# INIT GLOBALS
game_done = False
player_1 = 1
player_2 = -1
currentPlayer = 1
human_playing = True
game = None
network = None


# INIT GAME
def initGame():
    global game, network

    try:
        game = Game()
        network = Player2(game = game, networkName = Nets.none) # Change networkName to name of network to be used, none == random move
    except Exception as e:
        print(e)
        exit()


# PLAY GAME
def playGame():
    global currentPlayer, human_playing

    print("Welcome to Connect Four!")
    if args.second:
        currentPlayer = -1
    if args.aiRandom:
        human_playing = False

    # play connect four against the computer
    while not game_done:
        if game.check_status() is not None:
            print("Game is over!")
            print(np.flip(game.board, 0))
            break
        # If game still going, move
        move()
                
    # Show results
    showResults()


# PRINT THE GAMEBOARD
def printBoard():
    length = ""
    for i in range(1, game.width):
        length += "__" + str(i)
    print("___0" + length + "__")
    print(np.flip(game.board, 0))


# DO A MOVE
def move():
    # Show board
    printBoard()

    # If player is person, ask input
    if (currentPlayer == player_1 and human_playing):
        print("Player 1, please enter a column number: ")
        column = int(input())
        while (not checkMove(column)):
            print("Invalid move, try again")
            column = int(input())
        game.play_move(currentPlayer, column)

    # Computer move
    else:
        if (currentPlayer == player_1):
            print("Computer 1 is playing...")
        else:
            print("Computer 2 is playing...")

        # Let computer do move
        try:
            move = network.getMove()
            print("Computer played:", move)
            game.play_move(currentPlayer, move)
        except Exception as e:
            print(e)
            exit()

    # After move, change player
    changePlayer()


# CHECK IF PLAYER INPUT IS LEGAL
def checkMove(col):
    # Check width
    if ((col < 0) or (col >= game.width)):
        return False

    # Check height
    if (not game.is_legal_move(col)):
        return False

    return True


# CHANGE CURRENT PLAYER
def changePlayer():
    global currentPlayer

    if (currentPlayer == player_1):
        currentPlayer = player_2
    else:
        currentPlayer = player_1


# SHOW RESULTS OF THE GAME
def showResults():
    result = game.check_status()
    if (result == 0):
        print("It's a draw!")
    elif (result == player_1 and human_playing):
        print("You won!")
    elif (result == player_1):
        print("AI 1 won!")
    elif (human_playing):
        print("You lost!")
    else:
        print("Ai 2 won!")


def main():
    try:
        initGame()
        playGame()
    except KeyboardInterrupt as e:
        print("Exiting game early")

if __name__ == "__main__":
    main()
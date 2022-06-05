# Play the game against the computer or let the computer play against the computer.

from connectfour import Game, starting_player
from player2 import Player2, Nets
import argparse
import numpy as np


# INIT ARGS
parser = argparse.ArgumentParser(description='Play a game of connect four')
parser.add_argument('--first', metavar='S', nargs='?', default=False, const=True,
                    help='Use this if you want to go first')
parser.add_argument('--second', metavar='S', nargs='?', default=False, const=True,
                    help='Use this if you want to go second')
parser.add_argument('--aiRandom', metavar='A', nargs='?', default=False, const=True,
                    help='Use this if you want to have 2 random ais playing against each other')
args = parser.parse_args()

# INIT GLOBALS
game_done = False
player_1 = -1
player_2 = 1
currentPlayer = starting_player()
human_playing = True
game = None
network = None


# INIT GAME
def initGame():
    global game, network, currentPlayer

    try:
        game = Game()
        network = Player2(game = game, networkName = Nets.testNet) # Change networkName to name of network to be used, none == random move
    except Exception as e:
        print(e)
        exit()


# PLAY GAME
def playGame():
    global currentPlayer, human_playing

    print("Welcome to Connect Four!")
    if args.first:
        currentPlayer = -1
    if args.second:
        currentPlayer = 1
    if args.aiRandom:
        human_playing = False

    if (currentPlayer == player_1):
        print("Player 1 may play first")
    else:
        print("Player 2 may play first")

    # play connect four against the computer
    while not game_done:
        # Show board
        printBoard()

        # Check of game has ended
        if game.check_status() is not None:
            print("Game is over!")
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
    # If player is person, ask input
    if (currentPlayer == player_1):
        if (human_playing):
            print("Player 1, please enter a column number: ")
            column = askUserCol()
            while (not checkMove(column)):
                print("Invalid move, try again")
                column = askUserCol()
        # Random AI
        else: 
            print("Computer 1 is playing...")
            column = game.random_action()
            print("Computer 1 played:", column)

        game.play_move(currentPlayer, column)

    # Computer move
    else:
        print("Computer 2 is playing...")

        # Let computer do move
        try:
            move = network.getMove()
            print("Computer 2 played:", move)
            game.play_move(currentPlayer, move)
        except Exception as e:
            print(e)
            exit()

    # After move, change player
    changePlayer()


# ASK USER TO INPUT COL
def askUserCol():
    inp = input()
    while (not inp.isdigit()):
        print("No positive integer, try again")
        inp = input()
    return int(inp)


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
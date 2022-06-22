# Let random play against AI or random

from connectfour import Game, starting_player
from player2 import Player2, Nets
import numpy as np


# INIT GLOBALS
game_done = False
player_1 = -1
player_2 = 1
currentPlayer = starting_player()
human_playing = False
game = None
network = None

p2wins = 0
p2loss = 0
p2draw = 0

firstrun = True


# INIT GAME
def initGame():
    global game, network, currentPlayer, firstrun

    try:
        game = Game()
        if (firstrun == True):
            network = Player2(game = game, networkName = Nets.optimizedNet) # Change networkName to name of network to be used, none == random move
            firstrun = False
        else:
            network.setGame(game)

    except Exception as e:
        print(e)
        exit()


# PLAY GAME
def playGame():
    global currentPlayer, human_playing

    # print("Welcome to Connect Four!")

    # if (currentPlayer == player_1):
        # print("Player 1 may play first")
    # else:
        # print("Player 2 may play first")

    # play connect four against the computer
    while not game_done:
        # Show board
        # printBoard()

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
        # print("Computer 1 is playing...")
        column = game.random_action()
        # print("Computer 1 played:", column)

        game.play_move(currentPlayer, column)

    # Computer move
    else:
        # print("Computer 2 is playing...")

        # Let computer do move
        try:
            move = network.getMove()
            # print("Computer 2 played:", move)
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
    global p2wins, p2loss, p2draw

    result = game.check_status()
    if (result == 0):
        print("It's a draw!")
        p2draw += 1
    elif (result == player_1):
        print("AI 1 won!")
        p2loss += 1
    else:
        print("Ai 2 won!")
        p2wins += 1


def main():
    try:
        for i in range(100):
            initGame()
            print("-----",i,"-----")
            playGame()

            print("wins: ", p2wins)
            print("losses: ", p2loss)
            print("draws: ", p2draw)
            
    except KeyboardInterrupt as e:
        print("Exiting game early")

if __name__ == "__main__":
    main()
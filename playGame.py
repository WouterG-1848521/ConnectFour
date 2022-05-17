from connectfour import Game
import argparse

parser = argparse.ArgumentParser(description='Play a game of connect four')
parser.add_argument('--second', metavar='S', nargs='?', default=False, const=True,
                    help='Use this if you want to go second')
parser.add_argument('--aiRandom', metavar='A', nargs='?', default=False, const=True,
                    help='Use this if you want to have 2 random ais playing against each other')
args = parser.parse_args()

game = Game()


game_done = False
player_1 = 1
player_2 = -1
currentPlayer = 1
human_playing = True

print("Welcome to Connect Four!")
if args.second:
    currentPlayer = -1
if args.aiRandom:
    human_playing = False

# play connect four agains the computer using the random moves
while not game_done:
    if game.check_status() is not None:
        print("Game is over!")
        print(game)
        break
    if (currentPlayer == player_1 and human_playing):
        print(game)
        print("Player 1, please enter a column number: ")
        column = int(input())
        game.play_move(currentPlayer, column)
    elif (currentPlayer == player_1):
        print("Computer 1 is playing...")
        move = game.random_action(legal_only=True)
        print("Computer played:", move)
        game.play_move(currentPlayer, move)
    else:  
        print("Computer 2 is playing...")
        move = game.random_action(legal_only=True)
        print("Computer played:", move)
        game.play_move(currentPlayer, move)
    if (currentPlayer == player_1):
        currentPlayer = player_2
    else:
        currentPlayer = player_1

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
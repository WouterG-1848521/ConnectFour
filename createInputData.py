import numpy as np
import pandas as pd
from connectfour import Game
from convert import read_games_own, convert_games_to_setup
import csv

# creates input data for the neural network in the form of : 
# [ the best move it could make, a board state ]  (this best move is calculated by the winning function of the game class)

fname='c4-10k.csv'

# store the data in a csv file [best move, board state]
col_name = ['bestMove', '00','01','02','03','04','05','06',
                        '10','11','12','13','14','15','16',
                        '20','21','22','23','24','25','26',
                        '30','31','32','33','34','35','36',
                        '40','41','42','43','44','45','46',    
                        '50','51','52','53','54','55','56',] # (row, column)

with open("ProcessedData_withPlayer2.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(col_name)
    data = read_games_own(fname)
    counter = 0
    print("starting")
    for d in data:
        for i in d:
            writer.writerow(i)
            if (counter % 100 == 0):
                print("processed:", counter)
            counter += 1
            
print("done")
quit()
X = convert_games_to_setup(fname)

count = 0
for i in X:
    print(i)

x = read_games_own(fname)

data = pd.DataFrame(x, columns=col_name)

data.to_csv('computedData-100.csv', index=False)

print("done")

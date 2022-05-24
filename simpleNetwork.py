import tensorflow as tf

import numpy as np
import pandas as pd

from convert import convert_games_to_setup
from utils import convert_probablities_array_to_move

print("\n\n\n")

file_name = "ProcessedData copy.csv"
col_name = ['bestMove', '00','01','02','03','04','05','06',
                        '10','11','12','13','14','15','16',
                        '20','21','22','23','24','25','26',
                        '30','31','32','33','34','35','36',
                        '40','41','42','43','44','45','46',    
                        '50','51','52','53','54','55','56',] # (row, column)
DATA = pd.read_csv(file_name, names=col_name)
# DATA = convert_games_to_setup('c4-10k.csv')

train_features = []
train_labels = []
test_features = []
test_labels = []

total_data = 44350
amount_of_train_data = int(total_data / 2) 

# the first N rows are used for training
train_labels = DATA["bestMove"][:amount_of_train_data]
train_features = DATA[['00','01','02','03','04','05','06',
                     '10','11','12','13','14','15','16',
                     '20','21','22','23','24','25','26',
                     '30','31','32','33','34','35','36',
                     '40','41','42','43','44','45','46',    
                     '50','51','52','53','54','55','56']][:amount_of_train_data]
test_labels = DATA["bestMove"][amount_of_train_data: (2 * amount_of_train_data)]
test_features = DATA[['00','01','02','03','04','05','06',
                    '10','11','12','13','14','15','16',
                    '20','21','22','23','24','25','26',
                    '30','31','32','33','34','35','36',
                    '40','41','42','43','44','45','46',    
                    '50','51','52','53','54','55','56']][amount_of_train_data: (2 * amount_of_train_data)]

# counter = 0
# for it in DATA:
#     # for i in it:
#     if (counter < amount_of_train_data):
#         train_features.append([it[1:]])
#         train_labels.append(it[0])
#     else:   
#         test_features.append([it[1:]])
#         test_labels.append(it[0])
#     counter += 1
#     if (counter % (amount_of_train_data / 10) == 0):
#         print(counter)
#         # if (counter > (2 * amount_of_train_data)):
#         #     break
#     if (counter > (2 * amount_of_train_data)):
#         break

train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

print("Done loading data ",train_features.shape, " ", train_labels.shape, " ", test_features.shape, " ", test_labels.shape)
print("\n\n\n")

# for it in DATA: # THis should continue from the last iteration, but not sure yet (needs testing)
#     test_features = np.append(train_features, it[1:])
#     test_labels = np.append(train_labels, it[0])


# normalize ???? is dit nodig of niet

# idee : bord als input => move als output (deze zou de beste move moeten worden), kan gecheckt worden via smart move van game class

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    # change the 2d board to a 1D array
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # tf.nn.relu kinde as default activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    # testing with more layers => best 5
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_features,train_labels, epochs=10)

print(history.history)

print("Evaluate on test data")
results = model.evaluate(test_features, test_labels, batch_size=128)
print("test loss, test acc:", results)


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_features[:3])
print("predictions shape:", predictions.shape)
print("prediction: ", convert_probablities_array_to_move(predictions))
print("real labels: ", test_labels[:3])



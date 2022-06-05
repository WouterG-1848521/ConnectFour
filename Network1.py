from cProfile import label
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

total_data = 77451
amount_of_train_data = int(total_data / 4) * 3 

# the first N rows are used for training
train_labels = DATA["bestMove"][:amount_of_train_data]
train_features = DATA[['00','01','02','03','04','05','06',
                     '10','11','12','13','14','15','16',
                     '20','21','22','23','24','25','26',
                     '30','31','32','33','34','35','36',
                     '40','41','42','43','44','45','46',    
                     '50','51','52','53','54','55','56']][:amount_of_train_data]
test_labels = DATA["bestMove"][amount_of_train_data: ]
test_features = DATA[['00','01','02','03','04','05','06',
                    '10','11','12','13','14','15','16',
                    '20','21','22','23','24','25','26',
                    '30','31','32','33','34','35','36',
                    '40','41','42','43','44','45','46',    
                    '50','51','52','53','54','55','56']][amount_of_train_data: ]

train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

print("Done loading data ",train_features.shape, " ", train_labels.shape, " ", test_features.shape, " ", test_labels.shape)
print("\n\n\n")

# normalize ???? is dit nodig of niet

# idee : bord als input => move als output (deze zou de beste move moeten worden), kan gecheckt worden via smart move van game class

# results for 1 layer
# layer   | accuracy
# sigmoid | 0.2945
# relu    | 0.4729
# softmax | 0.3789
# tanh    | 0.0.4016

# results for size of layers
# size     | accuracy
# 1024     | 0.5171
# 512      | 0.5151
# 256      | 0.4980
# 128      | 0.4843
# 64       | 0.4623
# 32       | 0.4448
# 16       | 0.401
# 42       | 0.4556 (size of input)

# results when using regulatization
# type    | accuracy
# none    | 0.4623 (used size 64 of previous test)
# l1      | 0.2945
# l2      | 0.4312
# l1 + l2 | 0.2945
# OrthogonalRegularizer | 0.4678 (rows)
# OrthogonalRegularizer | 0.4704 (columns)
regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="columns")

# model.add(tf.keras.layers.Dropout(0.25)) # a dropout layer lowers the accurancy

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    # change the 2d board to a 1D array
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=regularizer))
# model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=10, min_delta=0.001)
early_stopping_monitor2 = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=10, min_delta=0.001)
history = model.fit(train_features,train_labels, epochs=10, callbacks=[early_stopping_monitor,early_stopping_monitor2])

print(history.history)

accuracies = history.history['accuracy']
losses = history.history['loss']
epochs = range(len(accuracies))

# show a plot of the accuracies and losses for training and testing
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax1.title.set_text('Accuracy')
ax2.title.set_text('Loss')

ax1.plot(epochs, accuracies, marker='o', color="b", label="Training accuracy")
# string = "Testing accuracy: " + str(accuracies[-1])
# ax1.annotate(string,xy=(len(epochs) / 2,accuracies[0-1] + 0.01))
ax2.plot(epochs, losses, marker='o', color="g", label="Training loss")
# string = "Testing loss: " + str(losses[0-1])
# ax2.annotate(string,xy=(len(epochs) / 2,losses[0-1] + 0.01))



print("Evaluate on test data")
results = model.evaluate(test_features, test_labels, batch_size=128)
print("test loss, test acc:", results)

testaccuracies = [results[1]] * len(accuracies)
testlosses = [results[0]] * len(accuracies)

ax1.plot(epochs, testaccuracies, color="r", label="Testing accuracy")
# string = "Testing accuracy: " + str(results[1])
# ax1.annotate(string,xy=(len(epochs) / 2,testaccuracies[0] + 0.01))
ax2.plot(epochs, testlosses, color="k", label="Testing loss")
# string = "Testing loss: " + str(results[0])
# ax1.annotate(string,xy=(len(epochs) / 2,testlosses[0] + 0.01))
fig.legend()

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_features[:3])
print("predictions shape:", predictions.shape)
print("prediction: ", convert_probablities_array_to_move(predictions))
print("real labels: ", test_labels[:3])


plt.show()
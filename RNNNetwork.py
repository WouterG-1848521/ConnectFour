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

# prepare data for training a recurrent neural network
# test 1
# dim_1 = 10
# dim_2 = int(len(train_features) / dim_1)
# dim_3 = 42

# leftover = len(train_features) % dim_1
# train_features = train_features[0: len(train_features) - leftover]
# train_features = train_features.reshape(dim_1, dim_2, dim_3)

# dim_2 = int(len(test_features) / dim_1)
# leftover = len(test_features) % dim_1
# test_features = test_features[0: len(test_features) - leftover]
# test_features = test_features.reshape(dim_1, dim_2, dim_3)

# test 2
# for i in range(len(train_features)):
#     np.insert(train_features, 0, i)
# for i in range(len(test_features)):
#     test_features[i] = [i, test_features[i]]
# inputs = np.random.random([32, 10, 8]).astype(np.float32)

# test 3
test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], 1)
train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], 1)

print("Done loading data ",train_features.shape, " ", train_labels.shape, " ", test_features.shape, " ", test_labels.shape)
print("\n\n\n")

# test a Recurrent Neural Network

regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="columns")

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.SimpleRNN(512, input_shape=(42, 1), activation='relu', kernel_regularizer=regularizer))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=10, min_delta=0.001)
early_stopping_monitor2 = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=10, min_delta=0.001)
history = model.fit(train_features, tf.keras.utils.to_categorical(train_labels), epochs=100,  callbacks=[early_stopping_monitor,early_stopping_monitor2])

print(history.history)

print("Evaluate on test data")
results = model.evaluate(test_features,  tf.keras.utils.to_categorical(test_labels), batch_size=128)
print("test loss, test acc:", results)


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_features[:3])
print("predictions shape:", predictions.shape)
print("prediction: ", convert_probablities_array_to_move(predictions))
print("real labels: ", test_labels[:3])



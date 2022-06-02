from neural_nets.network import Network
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import convert_probablities_array_to_move


class testNet(Network):
    train_labels = []
    train_features = []
    test_labels = []
    test_features = []
    
    def __init__(self):
        super().__init__("testNet")

        # init train and testdata
        self.initData()
        
        # Add layers to model
        self.addLayers()
        
        # Compile model
        self.model.compile( optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])


    # TRAIN THE MODEL
    def train(self):
        print("--------------------")
        print(self.name + ": training network")

        self.model.fit(self.train_features, self.train_labels, epochs=10)

        print(self.name + ": done training data")
        print("--------------------")


    # PREDICT MOVE
    def predict(self, board):
        print("--------------------")
        print(self.name + ": predicting move")

        predict = self.model.predict(np.array([board.flatten(),board.flatten()]))
        print(predict)
        convert = convert_probablities_array_to_move(predict)
        print("Convert to move:", convert)

        print(self.name + ": done predicting move")
        print("--------------------")

        return convert[0]


    # EVALUATE MODEL
    def evaluate(self):
        print("--------------------")
        print(self.name + ": evaluating model")

        results = self.model.evaluate(self.test_features, self.test_labels, batch_size=128)
        # print("test loss, test acc:", results)

        print(self.name + ": done evaluating model")
        print("--------------------")


        # OPENS DATAFILE AND CREATES TRAIN AND TEST DATA
    def initData(self):
        headers = ['bestMove',  '00','01','02','03','04','05','06',
                                '10','11','12','13','14','15','16',
                                '20','21','22','23','24','25','26',
                                '30','31','32','33','34','35','36',
                                '40','41','42','43','44','45','46',    
                                '50','51','52','53','54','55','56',]
        
        total_data = 44350
        amount_of_train_data = int(total_data / 2) 

        data = pd.read_csv("data/ProcessedData copy.csv", names = headers)
        self.train_labels = np.array(data["bestMove"][:amount_of_train_data])
        self.train_features = np.array(data[headers[1:]][:amount_of_train_data])
        self.test_labels = np.array(data["bestMove"][amount_of_train_data: (2 * amount_of_train_data)])
        self.test_features = np.array(data[headers[1:]][amount_of_train_data: (2 * amount_of_train_data)])


    # ADD DIFFERENT LAYERS TO MODEL
    def addLayers(self):
        self.model.add(tf.keras.layers.Flatten()) # change the 2d board to a 1D array

        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))

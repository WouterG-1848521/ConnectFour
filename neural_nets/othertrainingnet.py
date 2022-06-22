from neural_nets.network import Network
import tensorflow as tf
import pandas as pd
import numpy as np


class othertrainingNet(Network):
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
        
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=10, min_delta=0.001)
        early_stopping_monitor2 = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=10, min_delta=0.001)
        self.model.fit(self.train_features, self.train_labels, epochs=1, callbacks=[early_stopping_monitor,early_stopping_monitor2])


        print(self.name + ": done training data")
        print("--------------------")


    # PREDICT MOVE
    def predict(self, board):
        print("--------------------")
        print(self.name + ": predicting move")

        predict = self.model.predict(np.array([board.flatten(),board.flatten()]))
        print(predict)

        print(self.name + ": done predicting move")
        print("--------------------")

        return predict


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
                                '50','51','52','53','54','55','56', 'player']
        
        total_data = 54770
        amount_of_train_data = int(total_data / 4) * 3 

        data = pd.read_csv("data/ProcessedData_withPlayer_54000.csv", names = headers)
        self.train_labels = np.array(data["bestMove"][:amount_of_train_data])
        self.train_features = np.array(data[headers[1:-1]][:amount_of_train_data])
        self.test_labels = np.array(data["bestMove"][amount_of_train_data: ])
        self.test_features = np.array(data[headers[1:-1]][amount_of_train_data: ])


    # ADD DIFFERENT LAYERS TO MODEL
    def addLayers(self):
        self.model.add(tf.keras.layers.Flatten()) # change the 2d board to a 1D array

        regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="columns")

        self.model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizer))
        self.model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizer))
        self.model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizer))
        self.model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizer))
        self.model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizer))
        self.model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))

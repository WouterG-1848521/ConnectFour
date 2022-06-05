import tensorflow as tf


class Network():
    name = "[name of network]"
    model = None

    def __init__(self, name, initSequential = True):
        self.name = name

        if (initSequential == True):
            self.model = tf.keras.models.Sequential()


    def train(self):
        raise NotImplementedError("Method 'train()' not implemented in " + self.name)


    def predict(self):
        raise NotImplementedError("Method 'predict()' not implemented in " + self.name)


    def evaluate(self):
        raise NotImplementedError("Method 'evaluate()' not implemented in " + self.name)
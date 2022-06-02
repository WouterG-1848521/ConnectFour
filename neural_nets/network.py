class Network():
    name = "[name of network]"

    def __init__(self, name):
        self.name = name


    def train(self):
        raise NotImplementedError("Method 'train()' not implemented in " + self.name)


    def predict(self):
        raise NotImplementedError("Method 'predict()' not implemented in " + self.name)
from neural_nets.network import Network


class testNet(Network):
    def __init__(self):
        super().__init__("testNet")

    def train(self):
        print(self.name + ": training network")

    def predict(self):
        print(self.name + ": predicting move")
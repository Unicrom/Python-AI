from neuralNetworkBase import nueralNetwork

class AI:
    def __init__(self, parameters, id) -> None:
        self.nueralNetwork = nueralNetwork(parameters)
        self.id = id
    
    def randomize(self):
        return
    
    def decision(self, inputs):
        return
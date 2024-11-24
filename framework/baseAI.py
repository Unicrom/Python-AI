from neuralNetworkBase import neuralNetwork

class AI:
    def __init__(self, parameters, id) -> None:
        self.neuralNetwork = neuralNetwork(parameters)
        self.id = id
    
    def randomize(self):
        return
    
    def decision(self, inputs):
        return
from copy import deepcopy

class AI:
    def __init__(self, matrix, id) -> None:
        self.nueralNetwork = matrix.deepcopy()
        self.id = id
    
    def randomize(self):
        return
    
    def decision(self, inputs):
        return
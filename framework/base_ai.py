from framework.neural_network import NeuralNetwork

class BaseAI:
    def __init__(self, parameters: list, id:int) -> None:
        '''
        ### Creates an AI object
        Requires **parameters** and **id** as inputs
        
        ---
        **parameters:** *[list of integers]* each value in list represents the # of nodes in the respective layer\n
        **id:** *[int]* acts as the AI specifier, a Name
        '''
        self.neural_network = NeuralNetwork(parameters)
        self.id = id
    
    def randomize(self):
        return
    
    def decision(self, inputs):
        return
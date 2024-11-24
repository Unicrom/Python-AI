from framework.neural_network import NeuralNetwork


class BaseAI:
    def __init__(
        self, parameters: list, id: int, rand_values: bool, bias_range: float
    ) -> None:
        """
        **parameters:** *[list of int]* each value in list represents the # of nodes in the respective layer\n
        **id:** *[int]* acts as the AI specifier, a Name\n
        **rand_values:** *[bool]* if **True** creates the Neural Network with random values for the Weights and Biases
        **bias_range:** *[float]* possible range for the bias of each node **WHEN RANDOMLY GENERATED** (*max* = 0 + value, *min* = 0 - value)
        """
        self.neural_network = NeuralNetwork(parameters, rand_values, bias_range)
        self.id = id

    def randomize(self, bias_range: float = 3) -> None:
        """
        Randomizes the Weights and Biases of each Node of each Hidden layer\n
        **bias_range:** *[float]* possible range for the bias of each node (*max* = 0 + value, *min* = 0 - value)
        """
        self.neural_network.randomize(bias_range)

    def decision(self, inputs):
        return

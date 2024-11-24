from framework.layer import Layer


class NeuralNetwork:
    def __init__(
        self, parameters: list, rand_values: bool = True, bias_range: float = 3
    ) -> None:
        """
        *each value of **parameters** represents the number of nodes in the layer*\n
        > Creates **Input** and **Output** layers based on the first and last index of **parameters** respectively\n
        > Creates **Hidden** layers based on the remaining values within **parameters**\n
        ---
        **rand_values:** *[bool]* if **True** sets the weights and biases of each hidden layer node to a random value **Else** sets all to 0\n
        **bias_range:** *[float]* possible range for the bias of each node **WHEN RANDOMLY GENERATED** (*max* = 0 + value, *min* = 0 - value)
        """
        # Creates a list storing the Hidden Layers
        self.hidden_layers = [
            Layer(layer_size, rand_values, bias_range) # New basic Layer
            for layer_size in parameters[1:-1] # Loops through the values of parameters excluding the first and last item.
        ]

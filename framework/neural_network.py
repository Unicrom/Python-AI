from framework.layer import InputLayer, HiddenLayer, OutputLayer


class NeuralNetwork:
    def __init__(self, parameters: list, rand_values: bool, bias_range: float) -> None:
        """
        *each value of **parameters** represents the number of nodes in the layer*\n
        > Creates **Input** and **Output** layers based on the first and last index of **parameters** respectively\n
        > Creates **Hidden** layers based on the remaining values within **parameters**\n
        ---
        **rand_values:** *[bool]* if **True** sets the weights and biases of each hidden layer node to a random value **Else** sets all to 0\n
        **bias_range:** *[float]* possible range for the bias of each node **WHEN RANDOMLY GENERATED** (*max* = 0 + value, *min* = 0 - value)
        """
        self.input_layer = InputLayer(parameters[0]) # Creates InputLayer based on first item in parameters list
        self.input_layer.create_nodes()

        # Creates a list storing the Hidden Layers
        self.hidden_layers = [
            HiddenLayer(layer_size)  # New Hidden Layer
            for layer_size in parameters[
                1:-1
            ]  # Loops through the values (layer size) of parameters excluding the first and last item.
        ]
        for layer in self.hidden_layers: # Creates the Nodes for each Hidden Layer
            layer.create_nodes(rand_values, bias_range)

        self.output_layer = OutputLayer(parameters[-1]) # Creates InputLayer based on last item in parameters list
        self.output_layer.create_nodes(rand_values, bias_range)
    def randomize(self, bias_range: float) -> None:
        """
        Randomizes the Weights and Biases of each Node of each Hidden layer\n
        **bias_range:** *[float]* possible range for the bias of each node (*max* = 0 + value, *min* = 0 - value)
        """
        for layer in self.hidden_layers:
            layer.randomize_nodes(bias_range)

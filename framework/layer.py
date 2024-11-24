from framework.node import Node
from random import random


class Layer:
    def __init__(self, size: int, rand_values: bool, bias_range: float) -> None:
        """
        **size:** *[int]* number of nodes in layer\n
        **rand_values:** *[bool]* if **True** randomly generate values for the Weight and Bias of each node **Else** sets both to 0\n
        **bias_range:** *[float]* possible range for the bias of each node **WHEN RANDOMLY GENERATED** (*max* = 0 + value, *min* = 0 - value)
        """
        self.size = size
        self.nodes = [
            Node(0, 0) for _ in range(size)
        ]  # Creates list of Nodes with weight = 0 and bias = 0

        if rand_values:
            self.randomize_nodes(bias_range)  # If specified, randomize the Node values

    def randomize_nodes(self, bias_range: float):
        """
        #### Randomizes the **Weights** and **Biases** of each node\n
        **bias_range:** the *Min* and *Max* value of the new random Bias
        """
        for node in self.nodes:
            rand_weight = round(
                random() * 2 - 1, 4
            )  # Random value between -1 and 1 (Rounded to the thousandth)
            rand_bias = round(
                random() * (bias_range * 2) - bias_range, 4
            )  # Random value between -bias_range and bias_range (Rounded to the thousandth)

            node.set_weight(rand_weight)
            node.set_bias(rand_bias)

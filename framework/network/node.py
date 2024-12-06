import math


class Node:
    def __init__(self, weight: float, bias: float) -> None:
        """
        **weight:** the weight of this Node\n
        **bias:** the bias of this Node
        """
        self.weight = weight
        self.bias = bias
        self.value = 0
        self.value = None

    def set_weight(self, new_weight: bool) -> None:
        """
        Sets the **Weight** of the node to a new value
        """
        self.weight = new_weight

    def set_bias(self, new_bias: bool) -> None:
        """
        Sets the **Bias** of the node to a new value
        """
        self.bias = new_bias

    def calculate_value(self, previous_layer_sum: list) -> None:
        """Calculates the value of the Node based on the previous layer Nodes
        Args:
            previous_node_values (list): a list containing the values of each Node in the previous layer
        """
        self.value = self.sigmoid(previous_layer_sum)  - self.bias

    def sigmoid(self, x: float) -> float:
        """calculates the sigmoid of x

        Args:
            x (float): value being calculated

        Returns:
            float: returns the sigmoid of x
        """
        return 1 / (1 + math.exp(-x))
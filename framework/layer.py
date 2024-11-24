from framework.node import Node
from random import random


class Layer:
    def __init__(self, size: int) -> None:
        """
        **size:** *[int]* number of nodes in layer
        """
        self.size = size


class InputLayer(Layer):
    def create_nodes(self):
        """
        Creates the input Nodes (Does not have weigh or bias)"""
        self.inputs = [
            0 for _ in range(self.size)
        ]  # Creates a list of 0 based on number of inputs (size)

    def set_inputs(self, values: list) -> None:
        """
        Sets the values of the input nodes to the respective value in the values parameter\n
        *Throws Error if list sizes do not match*"""
        if not len(self.inputs) == len(values):
            raise Exception("Input Layer Size and new Values size do not match")
        self.inputs = values


class HiddenLayer(Layer):
    def create_nodes(self, rand_values: bool, bias_range: float) -> None:
        """
        **rand_values:** *[bool]* if **True** randomly generate values for the Weight and Bias of each node **Else** sets both to 0\n
        **bias_range:** *[float]* possible range for the bias of each node **WHEN RANDOMLY GENERATED** (*max* = 0 + value, *min* = 0 - value)
        """
        self.nodes = [
            Node(0, 0) for _ in range(self.size)
        ]  # Creates list of Nodes with weight = 0 and bias = 0

        if rand_values:
            self.randomize_nodes(bias_range)  # If specified, randomize the Node values

    def randomize_nodes(self, bias_range: float) -> None:
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


class OutputLayer(HiddenLayer):
    def get_top_node(self) -> int:
        """
        Returns the index of which Node has the highest value"""
        highest_value = 0
        for i, node in enumerate(self.nodes):  # Loops through Output Nodes
            if highest_value > node.value:
                continue  # Goes to the next iteration if value of node not higher not the greatest

            highest_value = node.value
            highest_value_index = i

        return highest_value_index

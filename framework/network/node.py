class Node:
    def __init__(self, weight: float, bias: float) -> None:
        """
        **weight:** the weight of this Node\n
        **bias:** the bias of this Node
        """
        self.weight = weight
        self.bias = bias
        self.value = 0

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

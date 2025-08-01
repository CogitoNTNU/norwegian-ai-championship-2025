import random
import math
from typing import Dict, List
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the network"""

    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class ActivationFunction(Enum):
    """Available activation functions"""

    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"


class Node:
    """
    Represents a node (neuron) in the NEAT network
    """

    def __init__(
        self,
        node_id: int,
        node_type: NodeType,
        activation: ActivationFunction = ActivationFunction.SIGMOID,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.activation = activation
        self.bias = 0.0
        self.incoming_connections: List["Connection"] = []
        self.outgoing_connections: List["Connection"] = []

    def activate(self, input_val: float) -> float:
        """
        Apply activation function to the input value

        Args:
            input_val: Input value to activate

        Returns:
            Activated output value
        """
        if self.activation == ActivationFunction.SIGMOID:
            return 1.0 / (1.0 + math.exp(-max(-500, min(500, input_val))))
        elif self.activation == ActivationFunction.TANH:
            return math.tanh(input_val)
        elif self.activation == ActivationFunction.RELU:
            return max(0, input_val)
        elif self.activation == ActivationFunction.LINEAR:
            return input_val

    def set_bias(self, bias: float):
        """
        Set the bias for this node

        Args:
            bias: Bias value to set
        """
        self.bias = bias

    def add_incoming_connection(self, connection: "Connection"):
        """Add an incoming connection to this node"""
        self.incoming_connections.append(connection)

    def add_outgoing_connection(self, connection: "Connection"):
        """Add an outgoing connection from this node"""
        self.outgoing_connections.append(connection)

    def __str__(self):
        return f"Node(id={self.node_id}, type={self.node_type.value}, activation={self.activation.value})"


class Connection:
    """
    Represents a connection (synapse) between two nodes in the NEAT network
    """

    def __init__(
        self,
        innovation_number: int,
        input_node: Node,
        output_node: Node,
        weight: float = None,
        enabled: bool = True,
    ):
        self.innovation_number = (
            innovation_number  # Unique identifier for this connection type
        )
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight if weight is not None else random.uniform(-1.0, 1.0)
        self.enabled = enabled

        # Add this connection to the nodes
        input_node.add_outgoing_connection(self)
        output_node.add_incoming_connection(self)

    def __str__(self):
        status = "enabled" if self.enabled else "disabled"
        return (
            f"Connection(innovation={self.innovation_number}, "
            f"{self.input_node.node_id}->{self.output_node.node_id}, "
            f"weight={self.weight:.3f}, {status})"
        )


class InnovationTracker:
    """
    Tracks innovation numbers for connections to ensure consistent evolution
    """

    def __init__(self):
        self.current_innovation = 0
        self.innovation_history: Dict[tuple, int] = {}

    def get_innovation_number(self, input_node_id: int, output_node_id: int) -> int:
        """
        Get innovation number for a connection between two nodes

        Args:
            input_node_id: ID of input node
            output_node_id: ID of output node

        Returns:
            Innovation number for this connection
        """
        connection_key = (input_node_id, output_node_id)

        if connection_key in self.innovation_history:
            return self.innovation_history[connection_key]

        self.current_innovation += 1
        self.innovation_history[connection_key] = self.current_innovation
        return self.current_innovation

    def reset(self):
        """Reset innovation tracking"""
        self.current_innovation = 0
        self.innovation_history.clear()


# Global innovation tracker instance
innovation_tracker = InnovationTracker()


if __name__ == "__main__":
    # Test the classes
    print("Testing NEAT Node and Connection classes...")

    # Create nodes
    input_node = Node(1, NodeType.INPUT)
    hidden_node = Node(2, NodeType.HIDDEN, ActivationFunction.SIGMOID)
    output_node = Node(3, NodeType.OUTPUT, ActivationFunction.SIGMOID)

    print(f"Created nodes: {input_node}, {hidden_node}, {output_node}")

    # Create connections
    conn1 = Connection(
        innovation_tracker.get_innovation_number(1, 2),
        input_node,
        hidden_node,
        weight=0.5,
    )
    conn2 = Connection(
        innovation_tracker.get_innovation_number(2, 3),
        hidden_node,
        output_node,
        weight=-0.3,
    )

    print(f"Created connections: {conn1}, {conn2}")

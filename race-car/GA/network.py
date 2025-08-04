import random
import math
from typing import Dict, List, Optional
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


class Network:
    """
    Represents a NEAT neural network with nodes and connections
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        self.nodes: Dict[int, Node] = {}
        self.connections: Dict[int, Connection] = {}
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self.hidden_nodes: List[Node] = []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.next_node_id = 0
        self.fitness = -1.0  # Default fitness value

        # Create input and output nodes
        self._create_initial_nodes()

    def _create_initial_nodes(self):
        """Create the initial input and output nodes"""
        # Create input nodes
        for i in range(self.num_inputs):
            node = Node(self.next_node_id, NodeType.INPUT)
            self.nodes[self.next_node_id] = node
            self.input_nodes.append(node)
            self.next_node_id += 1

        # Create output nodes
        for i in range(self.num_outputs):
            node = Node(self.next_node_id, NodeType.OUTPUT)
            self.nodes[self.next_node_id] = node
            self.output_nodes.append(node)
            self.next_node_id += 1

    def add_connection(
        self, input_node_id: int, output_node_id: int, weight: float = None
    ) -> Optional[Connection]:
        """
        Add a connection between two nodes

        Args:
            input_node_id: ID of the input node
            output_node_id: ID of the output node
            weight: Weight of the connection (random if None)

        Returns:
            The created connection or None if invalid
        """
        if input_node_id not in self.nodes or output_node_id not in self.nodes:
            return None

        # Check if connection already exists
        for conn in self.connections.values():
            if (
                conn.input_node.node_id == input_node_id
                and conn.output_node.node_id == output_node_id
            ):
                return None  # Connection already exists

        # Prevent cycles (simplified check)
        if self._would_create_cycle(input_node_id, output_node_id):
            return None

        innovation_num = innovation_tracker.get_innovation_number(
            input_node_id, output_node_id
        )
        connection = Connection(
            innovation_num,
            self.nodes[input_node_id],
            self.nodes[output_node_id],
            weight,
        )

        self.connections[innovation_num] = connection
        return connection

    def add_node(self, connection_to_split: Connection) -> Optional[Node]:
        """
        Add a new hidden node by splitting an existing connection

        Args:
            connection_to_split: The connection to split with a new node

        Returns:
            The new node or None if invalid
        """
        if connection_to_split.innovation_number not in self.connections:
            return None

        # Disable the old connection
        connection_to_split.enabled = False

        # Create new hidden node
        new_node = Node(self.next_node_id, NodeType.HIDDEN)
        self.nodes[self.next_node_id] = new_node
        self.hidden_nodes.append(new_node)
        self.next_node_id += 1

        # Create two new connections
        # Connection from input to new node (weight = 1.0)
        self.add_connection(
            connection_to_split.input_node.node_id, new_node.node_id, 1.0
        )

        # Connection from new node to output (weight = original weight)
        self.add_connection(
            new_node.node_id,
            connection_to_split.output_node.node_id,
            connection_to_split.weight,
        )

        return new_node

    def _would_create_cycle(self, input_node_id: int, output_node_id: int) -> bool:
        """
        Check if adding a connection would create a cycle

        Args:
            input_node_id: ID of the input node
            output_node_id: ID of the output node

        Returns:
            True if cycle would be created, False otherwise
        """
        # Simple cycle detection: check if output_node can reach input_node
        visited = set()

        def can_reach(current_id: int, target_id: int) -> bool:
            if current_id == target_id:
                return True
            if current_id in visited:
                return False

            visited.add(current_id)

            # Find all outgoing connections from current node
            for conn in self.connections.values():
                if (
                    conn.enabled
                    and conn.input_node.node_id == current_id
                    and can_reach(conn.output_node.node_id, target_id)
                ):
                    return True

            return False

        return can_reach(output_node_id, input_node_id)

    def _topological_sort(self) -> List[Node]:
        """
        Get nodes in topological order for forward pass

        Returns:
            List of nodes in topological order
        """
        visited = set()
        temp_visited = set()
        result = []

        def visit(node_id: int):
            if node_id in temp_visited:
                return  # Cycle detected, skip
            if node_id in visited:
                return

            temp_visited.add(node_id)

            # Visit all nodes that this node connects to
            for conn in self.connections.values():
                if conn.enabled and conn.input_node.node_id == node_id:
                    visit(conn.output_node.node_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(self.nodes[node_id])

        # Start with input nodes
        for node in self.input_nodes:
            visit(node.node_id)

        # Then visit any remaining nodes
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)

        return result

    def forward_pass(self, inputs: List[float]) -> List[float]:
        """
        Perform forward pass through the network

        Args:
            inputs: List of input values

        Returns:
            List of output values
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")

        # Store node outputs during this forward pass
        node_outputs = {}

        # Set input node outputs
        for i, input_node in enumerate(self.input_nodes):
            node_outputs[input_node.node_id] = inputs[i]

        # Get nodes in topological order
        sorted_nodes = self._topological_sort()

        # Evaluate each node
        for node in sorted_nodes:
            if node.node_type == NodeType.INPUT:
                continue  # Already set

            # Sum all incoming signals
            total_input = node.bias  # Start with bias
            for conn in self.connections.values():
                if (
                    conn.enabled
                    and conn.output_node.node_id == node.node_id
                    and conn.input_node.node_id in node_outputs
                ):
                    # Calculate signal directly: input_value * weight
                    signal = node_outputs[conn.input_node.node_id] * conn.weight
                    total_input += signal

            # Activate and store output
            node_outputs[node.node_id] = node.activate(total_input)

        # Return output node values
        return [node_outputs.get(node.node_id, 0.0) for node in self.output_nodes]

    def get_complexity(self) -> Dict[str, int]:
        """
        Get network complexity metrics

        Returns:
            Dictionary with complexity metrics
        """
        enabled_connections = sum(
            1 for conn in self.connections.values() if conn.enabled
        )

        return {
            "nodes": len(self.nodes),
            "connections": len(self.connections),
            "enabled_connections": enabled_connections,
            "hidden_nodes": len(self.hidden_nodes),
        }

    def __str__(self):
        complexity = self.get_complexity()
        return f"Network(nodes={complexity['nodes']}, connections={complexity['enabled_connections']})"


if __name__ == "__main__":
    # Test the Network class
    print("Testing NEAT Network class...")

    # Create a simple network
    network = Network(num_inputs=2, num_outputs=1)
    print(f"Created network: {network}")

    # Add some connections
    # Connect input 0 to output 0
    conn1 = network.add_connection(0, 2, weight=0.5)
    # Connect input 1 to output 0
    conn2 = network.add_connection(1, 2, weight=-0.3)

    print(f"Added connections: {conn1}, {conn2}")

    # Test forward pass
    inputs = [1.0, 0.5]
    outputs = network.forward_pass(inputs)
    print(f"Forward pass: inputs={inputs} -> outputs={outputs}")

    # Add a hidden node by splitting a connection
    if conn1:
        new_node = network.add_node(conn1)
        print(f"Added hidden node: {new_node}")
        print(f"Network complexity: {network.get_complexity()}")

        # Test forward pass with hidden node
        outputs = network.forward_pass(inputs)
        print(f"Forward pass with hidden node: inputs={inputs} -> outputs={outputs}")

    print("Network testing complete!")

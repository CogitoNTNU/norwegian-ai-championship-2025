import pytest
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "DM-i-AI-2025" / "race-car" / "GA"))

from network import (
    Node,
    Connection,
    Network,
    InnovationTracker,
    NodeType,
    ActivationFunction,
    innovation_tracker,
)


class TestNode:
    """Test cases for the Node class"""

    def test_node_creation(self):
        """Test basic node creation"""
        node = Node(1, NodeType.INPUT)
        assert node.node_id == 1
        assert node.node_type == NodeType.INPUT
        assert node.activation == ActivationFunction.SIGMOID
        assert node.bias == 0.0

    def test_node_creation_with_activation(self):
        """Test node creation with specific activation function"""
        node = Node(2, NodeType.HIDDEN, ActivationFunction.RELU)
        assert node.node_id == 2
        assert node.node_type == NodeType.HIDDEN
        assert node.activation == ActivationFunction.RELU

    def test_sigmoid_activation(self):
        """Test sigmoid activation function"""
        node = Node(1, NodeType.HIDDEN, ActivationFunction.SIGMOID)

        # Test typical values
        assert abs(node.activate(0.0) - 0.5) < 1e-6
        assert node.activate(-1000) < 0.01  # Large negative
        assert node.activate(1000) > 0.99  # Large positive

        # Test specific case
        result = node.activate(1.0)
        expected = 1.0 / (1.0 + math.exp(-1.0))
        assert abs(result - expected) < 1e-6

    def test_tanh_activation(self):
        """Test tanh activation function"""
        node = Node(1, NodeType.HIDDEN, ActivationFunction.TANH)

        assert abs(node.activate(0.0) - 0.0) < 1e-6
        assert abs(node.activate(1.0) - math.tanh(1.0)) < 1e-6
        assert node.activate(-1000) < -0.99
        assert node.activate(1000) > 0.99

    def test_relu_activation(self):
        """Test ReLU activation function"""
        node = Node(1, NodeType.HIDDEN, ActivationFunction.RELU)

        assert node.activate(0.0) == 0.0
        assert node.activate(-1.0) == 0.0
        assert node.activate(5.0) == 5.0
        assert node.activate(-100.0) == 0.0

    def test_linear_activation(self):
        """Test linear activation function"""
        node = Node(1, NodeType.HIDDEN, ActivationFunction.LINEAR)

        assert node.activate(0.0) == 0.0
        assert node.activate(5.0) == 5.0
        assert node.activate(-3.0) == -3.0
        assert node.activate(100.0) == 100.0

    def test_set_bias(self):
        """Test setting node bias"""
        node = Node(1, NodeType.HIDDEN)
        node.set_bias(0.5)
        assert node.bias == 0.5

        node.set_bias(-1.0)
        assert node.bias == -1.0

    def test_node_string_representation(self):
        """Test node string representation"""
        node = Node(1, NodeType.INPUT)
        expected = "Node(id=1, type=input, activation=sigmoid)"
        assert str(node) == expected


class TestConnection:
    """Test cases for the Connection class"""

    def test_connection_creation(self):
        """Test basic connection creation"""
        input_node = Node(1, NodeType.INPUT)
        output_node = Node(2, NodeType.OUTPUT)

        conn = Connection(1, input_node, output_node, weight=0.5)

        assert conn.innovation_number == 1
        assert conn.input_node == input_node
        assert conn.output_node == output_node
        assert conn.weight == 0.5
        assert conn.enabled

    def test_connection_creation_random_weight(self):
        """Test connection creation with random weight"""
        input_node = Node(1, NodeType.INPUT)
        output_node = Node(2, NodeType.OUTPUT)

        conn = Connection(1, input_node, output_node)

        assert -1.0 <= conn.weight <= 1.0
        assert conn.enabled

    def test_connection_disabled(self):
        """Test creating disabled connection"""
        input_node = Node(1, NodeType.INPUT)
        output_node = Node(2, NodeType.OUTPUT)

        conn = Connection(1, input_node, output_node, enabled=False)
        assert not conn.enabled

    def test_connection_string_representation(self):
        """Test connection string representation"""
        input_node = Node(1, NodeType.INPUT)
        output_node = Node(2, NodeType.OUTPUT)

        conn = Connection(1, input_node, output_node, weight=0.5)
        expected = "Connection(innovation=1, 1->2, weight=0.500, enabled)"
        assert str(conn) == expected

        conn.enabled = False
        expected = "Connection(innovation=1, 1->2, weight=0.500, disabled)"
        assert str(conn) == expected


class TestInnovationTracker:
    """Test cases for the InnovationTracker class"""

    def test_innovation_tracker_creation(self):
        """Test basic innovation tracker creation"""
        tracker = InnovationTracker()
        assert tracker.current_innovation == 0
        assert len(tracker.innovation_history) == 0

    def test_get_innovation_number_new(self):
        """Test getting innovation number for new connection"""
        tracker = InnovationTracker()

        innovation = tracker.get_innovation_number(1, 2)
        assert innovation == 1
        assert tracker.current_innovation == 1
        assert (1, 2) in tracker.innovation_history

    def test_get_innovation_number_existing(self):
        """Test getting innovation number for existing connection"""
        tracker = InnovationTracker()

        # First call
        innovation1 = tracker.get_innovation_number(1, 2)
        assert innovation1 == 1

        # Second call with same nodes
        innovation2 = tracker.get_innovation_number(1, 2)
        assert innovation2 == 1  # Should be same
        assert tracker.current_innovation == 1  # Should not increment

    def test_multiple_innovations(self):
        """Test multiple different innovations"""
        tracker = InnovationTracker()

        innovation1 = tracker.get_innovation_number(1, 2)
        innovation2 = tracker.get_innovation_number(2, 3)
        innovation3 = tracker.get_innovation_number(1, 3)

        assert innovation1 == 1
        assert innovation2 == 2
        assert innovation3 == 3
        assert tracker.current_innovation == 3

    def test_reset_tracker(self):
        """Test resetting innovation tracker"""
        tracker = InnovationTracker()

        tracker.get_innovation_number(1, 2)
        tracker.get_innovation_number(2, 3)

        assert tracker.current_innovation == 2
        assert len(tracker.innovation_history) == 2

        tracker.reset()

        assert tracker.current_innovation == 0
        assert len(tracker.innovation_history) == 0


class TestNetwork:
    """Test cases for the Network class"""

    def test_network_creation(self):
        """Test basic network creation"""
        network = Network(num_inputs=2, num_outputs=1)

        assert network.num_inputs == 2
        assert network.num_outputs == 1
        assert len(network.input_nodes) == 2
        assert len(network.output_nodes) == 1
        assert len(network.hidden_nodes) == 0
        assert network.next_node_id == 3  # 2 inputs + 1 output

    def test_network_initial_nodes(self):
        """Test that initial nodes are created correctly"""
        network = Network(num_inputs=3, num_outputs=2)

        # Check input nodes
        assert len(network.input_nodes) == 3
        for i, node in enumerate(network.input_nodes):
            assert node.node_id == i
            assert node.node_type == NodeType.INPUT

        # Check output nodes
        assert len(network.output_nodes) == 2
        for i, node in enumerate(network.output_nodes):
            assert node.node_id == 3 + i
            assert node.node_type == NodeType.OUTPUT

    def test_add_connection_valid(self):
        """Test adding valid connection"""
        network = Network(num_inputs=2, num_outputs=1)

        conn = network.add_connection(0, 2, weight=0.5)

        assert conn is not None
        assert conn.input_node.node_id == 0
        assert conn.output_node.node_id == 2
        assert conn.weight == 0.5
        assert len(network.connections) == 1

    def test_add_connection_invalid_nodes(self):
        """Test adding connection with invalid nodes"""
        network = Network(num_inputs=2, num_outputs=1)

        # Non-existent input node
        conn1 = network.add_connection(99, 2)
        assert conn1 is None

        # Non-existent output node
        conn2 = network.add_connection(0, 99)
        assert conn2 is None

    def test_add_connection_duplicate(self):
        """Test adding duplicate connection"""
        network = Network(num_inputs=2, num_outputs=1)

        conn1 = network.add_connection(0, 2)
        conn2 = network.add_connection(0, 2)  # Duplicate

        assert conn1 is not None
        assert conn2 is None
        assert len(network.connections) == 1

    def test_add_node_valid(self):
        """Test adding node by splitting connection"""
        network = Network(num_inputs=2, num_outputs=1)

        # Add connection first
        conn = network.add_connection(0, 2, weight=0.5)
        assert conn is not None

        # Split the connection
        new_node = network.add_node(conn)

        assert new_node is not None
        assert new_node.node_type == NodeType.HIDDEN
        assert len(network.hidden_nodes) == 1
        assert not conn.enabled  # Original connection should be disabled

        # Should have 3 connections total (1 disabled + 2 new)
        assert len(network.connections) == 3

        # Check that new connections exist
        enabled_connections = [c for c in network.connections.values() if c.enabled]
        assert len(enabled_connections) == 2

    def test_forward_pass_simple(self):
        """Test simple forward pass without hidden nodes"""
        network = Network(num_inputs=2, num_outputs=1)

        # Add connections
        network.add_connection(0, 2, weight=0.5)  # input0 -> output0
        network.add_connection(1, 2, weight=0.3)  # input1 -> output0

        # Test forward pass
        inputs = [1.0, 2.0]
        outputs = network.forward_pass(inputs)

        assert len(outputs) == 1

        # Calculate expected output
        # total_input = 1.0 * 0.5 + 2.0 * 0.3 = 1.1
        # sigmoid(1.1) ≈ 0.7503
        expected = 1.0 / (1.0 + math.exp(-1.1))
        assert abs(outputs[0] - expected) < 1e-4

    def test_forward_pass_with_bias(self):
        """Test forward pass with bias"""
        network = Network(num_inputs=1, num_outputs=1)

        # Set bias on output node
        network.output_nodes[0].set_bias(0.5)

        # Add connection
        network.add_connection(0, 1, weight=1.0)

        # Test forward pass
        outputs = network.forward_pass([1.0])

        # total_input = 1.0 * 1.0 + 0.5 (bias) = 1.5
        expected = 1.0 / (1.0 + math.exp(-1.5))
        assert abs(outputs[0] - expected) < 1e-6

    def test_forward_pass_wrong_input_size(self):
        """Test forward pass with wrong input size"""
        network = Network(num_inputs=2, num_outputs=1)

        with pytest.raises(ValueError):
            network.forward_pass([1.0])  # Too few inputs

        with pytest.raises(ValueError):
            network.forward_pass([1.0, 2.0, 3.0])  # Too many inputs

    def test_forward_pass_with_hidden_node(self):
        """Test forward pass with hidden node"""
        network = Network(num_inputs=1, num_outputs=1)

        # Add initial connection
        conn = network.add_connection(0, 1, weight=0.5)

        # Split connection to add hidden node
        network.add_node(conn)

        # Test forward pass
        outputs = network.forward_pass([1.0])

        # Should work without errors
        assert len(outputs) == 1
        assert isinstance(outputs[0], float)

    def test_cycle_detection(self):
        """Test cycle detection"""
        network = Network(num_inputs=1, num_outputs=1)

        # Add hidden node
        conn = network.add_connection(0, 1, weight=0.5)
        hidden_node = network.add_node(conn)

        # Try to create a cycle (hidden -> input)
        cycle_conn = network.add_connection(hidden_node.node_id, 0)

        # Should fail due to cycle detection
        assert cycle_conn is None

    def test_disabled_connections(self):
        """Test that disabled connections don't affect forward pass"""
        network = Network(num_inputs=2, num_outputs=1)

        # Add connections
        conn1 = network.add_connection(0, 2, weight=1.0)
        conn2 = network.add_connection(1, 2, weight=1.0)

        # Test with both connections enabled
        outputs1 = network.forward_pass([1.0, 1.0])

        # Disable one connection
        conn1.enabled = False
        conn2.enabled = True

        # Test with one connection disabled
        outputs2 = network.forward_pass([1.0, 1.0])

        # Outputs should be different
        assert outputs1[0] != outputs2[0]

    def test_get_complexity(self):
        """Test network complexity calculation"""
        network = Network(num_inputs=2, num_outputs=1)

        # Initial complexity
        complexity = network.get_complexity()
        assert complexity["nodes"] == 3
        assert complexity["connections"] == 0
        assert complexity["enabled_connections"] == 0
        assert complexity["hidden_nodes"] == 0

        # Add connection
        conn = network.add_connection(0, 2)
        complexity = network.get_complexity()
        assert complexity["connections"] == 1
        assert complexity["enabled_connections"] == 1

        # Add hidden node
        network.add_node(conn)
        complexity = network.get_complexity()
        assert complexity["nodes"] == 4
        assert complexity["connections"] == 3
        assert complexity["enabled_connections"] == 2  # Original disabled
        assert complexity["hidden_nodes"] == 1

    def test_network_string_representation(self):
        """Test network string representation"""
        network = Network(num_inputs=2, num_outputs=1)
        network.add_connection(0, 2)

        network_str = str(network)
        assert "Network(nodes=3, connections=1)" == network_str


class TestIntegration:
    """Integration tests for the complete system"""

    def test_global_innovation_tracker(self):
        """Test that global innovation tracker works across networks"""
        # Reset global tracker
        innovation_tracker.reset()

        network1 = Network(num_inputs=2, num_outputs=1)
        network2 = Network(num_inputs=2, num_outputs=1)

        # Add same connection to both networks
        conn1 = network1.add_connection(0, 2)
        conn2 = network2.add_connection(0, 2)

        # Should have same innovation number
        assert conn1.innovation_number == conn2.innovation_number

    def test_complex_network_evolution(self):
        """Test a more complex network evolution scenario"""
        network = Network(num_inputs=3, num_outputs=2)

        # Add initial connections
        conn1 = network.add_connection(0, 3, weight=0.5)  # input0 -> output0
        conn2 = network.add_connection(1, 4, weight=-0.3)  # input1 -> output1
        conn3 = network.add_connection(2, 3, weight=0.8)  # input2 -> output0

        # Test initial forward pass
        outputs = network.forward_pass([1.0, 0.5, -0.5])
        assert len(outputs) == 2

        # Add hidden nodes
        hidden1 = network.add_node(conn1)
        hidden2 = network.add_node(conn2)
        conn3.enabled = True  # Disable original connection

        # Add cross connections
        network.add_connection(hidden1.node_id, 4, weight=0.2)  # hidden1 -> output1
        network.add_connection(hidden2.node_id, 3, weight=-0.1)  # hidden2 -> output0

        # Test complex forward pass
        outputs = network.forward_pass([1.0, 0.5, -0.5])
        assert len(outputs) == 2
        assert isinstance(outputs[0], float)
        assert isinstance(outputs[1], float)

        # Check network complexity
        complexity = network.get_complexity()
        assert complexity["nodes"] == 7  # 3 input + 2 output + 2 hidden
        assert complexity["hidden_nodes"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

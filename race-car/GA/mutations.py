import random
from network import Network, ActivationFunction


def set_mutation_seed(seed: int = None):
    """
    Set the random seed for reproducible mutations

    Args:
        seed: Random seed value. If None, uses current time
    """
    random.seed(seed)


class MutationConfig:
    """Configuration parameters for mutations"""

    def __init__(self):
        # Probability of each mutation type
        self.add_connection_rate = 0.2
        self.add_node_rate = 0.1
        self.weight_mutation_rate = 0.8
        self.bias_mutation_rate = 0.2
        self.disable_connection_rate = 0.01
        self.enable_connection_rate = 0.01

        # Weight mutation parameters
        self.weight_perturbation_rate = 0.9  # Probability of perturbing vs replacing
        self.weight_perturbation_strength = 0.1
        self.weight_replacement_range = 2.0

        # Bias mutation parameters
        self.bias_perturbation_rate = 0.9
        self.bias_perturbation_strength = 0.1
        self.bias_replacement_range = 1.0

        # Connection limits
        self.max_connection_attempts = 50


def mutate_add_connection(network: Network, config: MutationConfig) -> bool:
    """
    Attempt to add a new connection between two nodes

    Args:
        network: Network to mutate
        config: Mutation configuration

    Returns:
        True if connection was added successfully, False otherwise
    """
    if random.random() > config.add_connection_rate:
        return False

    # Get all possible connections (avoid cycles and duplicates)
    possible_connections = []

    # From input nodes to hidden/output nodes
    for input_node in network.input_nodes:
        for hidden_node in network.hidden_nodes + network.output_nodes:
            if not _connection_exists(network, input_node.node_id, hidden_node.node_id):
                possible_connections.append((input_node.node_id, hidden_node.node_id))

    # From hidden nodes to hidden/output nodes (avoiding cycles)
    for hidden_node in network.hidden_nodes:
        for target_node in network.hidden_nodes + network.output_nodes:
            if (
                hidden_node.node_id != target_node.node_id
                and not _connection_exists(
                    network, hidden_node.node_id, target_node.node_id
                )
                and not network._would_create_cycle(
                    hidden_node.node_id, target_node.node_id
                )
            ):
                possible_connections.append((hidden_node.node_id, target_node.node_id))

    if not possible_connections:
        return False

    # Select random connection
    input_id, output_id = random.choice(possible_connections)

    # Add the connection
    connection = network.add_connection(input_id, output_id)
    return connection is not None


def mutate_add_node(network: Network, config: MutationConfig) -> bool:
    """
    Attempt to add a new node by splitting an existing connection

    Args:
        network: Network to mutate
        config: Mutation configuration

    Returns:
        True if node was added successfully, False otherwise
    """
    if random.random() > config.add_node_rate:
        return False

    # Get all enabled connections
    enabled_connections = [
        conn for conn in network.connections.values() if conn.enabled
    ]

    if not enabled_connections:
        return False

    # Select random connection to split
    connection_to_split = random.choice(enabled_connections)

    # Add the node
    new_node = network.add_node(connection_to_split)
    return new_node is not None


def mutate_weights(network: Network, config: MutationConfig) -> bool:
    """
    Mutate the weights of connections in the network

    Args:
        network: Network to mutate
        config: Mutation configuration

    Returns:
        True if any weights were mutated, False otherwise
    """
    if random.random() > config.weight_mutation_rate:
        return False

    mutated = False

    for connection in network.connections.values():
        if not connection.enabled:
            continue

        if random.random() < config.weight_mutation_rate:
            if random.random() < config.weight_perturbation_rate:
                # Perturb existing weight
                perturbation = random.gauss(0, config.weight_perturbation_strength)
                connection.weight += perturbation
            else:
                # Replace with new random weight
                connection.weight = random.uniform(
                    -config.weight_replacement_range, config.weight_replacement_range
                )
            mutated = True

    return mutated


def mutate_biases(network: Network, config: MutationConfig) -> bool:
    """
    Mutate the biases of nodes in the network

    Args:
        network: Network to mutate
        config: Mutation configuration

    Returns:
        True if any biases were mutated, False otherwise
    """
    if random.random() > config.bias_mutation_rate:
        return False

    mutated = False

    # Only mutate hidden and output nodes (input nodes don't use bias)
    for node in network.hidden_nodes + network.output_nodes:
        if random.random() < config.bias_mutation_rate:
            if random.random() < config.bias_perturbation_rate:
                # Perturb existing bias
                perturbation = random.gauss(0, config.bias_perturbation_strength)
                node.bias += perturbation
            else:
                # Replace with new random bias
                node.bias = random.uniform(
                    -config.bias_replacement_range, config.bias_replacement_range
                )
            mutated = True

    return mutated


def mutate_enable_disable_connections(network: Network, config: MutationConfig) -> bool:
    """
    Randomly enable or disable connections

    Args:
        network: Network to mutate
        config: Mutation configuration

    Returns:
        True if any connections were toggled, False otherwise
    """
    mutated = False

    for connection in network.connections.values():
        if connection.enabled and random.random() < config.disable_connection_rate:
            connection.enabled = False
            mutated = True
        elif not connection.enabled and random.random() < config.enable_connection_rate:
            connection.enabled = True
            mutated = True

    return mutated


def mutate_activation_functions(network: Network, mutation_rate: float = 0.05) -> bool:
    """
    Mutate activation functions of hidden and output nodes

    Args:
        network: Network to mutate
        mutation_rate: Probability of mutating each node's activation function

    Returns:
        True if any activation functions were changed, False otherwise
    """
    if random.random() > mutation_rate:
        return False

    mutated = False
    activation_functions = list(ActivationFunction)

    # Only mutate hidden and output nodes
    for node in network.hidden_nodes + network.output_nodes:
        if random.random() < mutation_rate:
            # Choose a different activation function
            available_functions = [
                af for af in activation_functions if af != node.activation
            ]
            if available_functions:
                node.activation = random.choice(available_functions)
                mutated = True

    return mutated


def mutate_network(network: Network, config: MutationConfig = None) -> dict:
    """
    Apply all mutation types to a network

    Args:
        network: Network to mutate
        config: Mutation configuration (uses default if None)

    Returns:
        Dictionary with mutation results
    """
    if config is None:
        config = MutationConfig()

    results = {
        "add_connection": False,
        "add_node": False,
        "mutate_weights": False,
        "mutate_biases": False,
        "toggle_connections": False,
        "mutate_activations": False,
    }

    # Apply mutations
    results["add_connection"] = mutate_add_connection(network, config)
    results["add_node"] = mutate_add_node(network, config)
    results["mutate_weights"] = mutate_weights(network, config)
    results["mutate_biases"] = mutate_biases(network, config)
    results["toggle_connections"] = mutate_enable_disable_connections(network, config)
    results["mutate_activations"] = mutate_activation_functions(network)

    return results


def _connection_exists(network: Network, input_id: int, output_id: int) -> bool:
    """
    Check if a connection already exists between two nodes

    Args:
        network: Network to check
        input_id: Input node ID
        output_id: Output node ID

    Returns:
        True if connection exists, False otherwise
    """
    for connection in network.connections.values():
        if (
            connection.input_node.node_id == input_id
            and connection.output_node.node_id == output_id
        ):
            return True
    return False


if __name__ == "__main__":
    # Test mutations
    print("Testing NEAT mutations...")

    # Create a simple network
    network = Network(num_inputs=3, num_outputs=2)
    print(f"Initial network: {network}")
    print(f"Initial complexity: {network.get_complexity()}")

    # Add some initial connections
    network.add_connection(0, 3, weight=0.5)  # input 0 -> output 0
    network.add_connection(1, 4, weight=-0.3)  # input 1 -> output 1
    network.add_connection(2, 3, weight=0.8)  # input 2 -> output 0

    print(f"After initial connections: {network.get_complexity()}")

    # Test mutations
    config = MutationConfig()

    print("\nTesting mutations...")
    for i in range(10):
        print(f"\nMutation round {i + 1}:")
        results = mutate_network(network, config)
        print(f"Mutation results: {results}")
        print(f"Network complexity: {network.get_complexity()}")

        # Test forward pass to ensure network still works
        inputs = [0.5, -0.2, 0.8]
        try:
            outputs = network.forward_pass(inputs)
            print(f"Forward pass successful: {inputs} -> {outputs}")
        except Exception as e:
            print(f"Forward pass failed: {e}")

    print("\nMutation testing complete!")

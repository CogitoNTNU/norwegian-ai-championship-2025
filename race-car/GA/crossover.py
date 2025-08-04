import random
from typing import Dict, Optional
from network import Network, Node, Connection, NodeType, innovation_tracker


class CrossoverConfig:
    """Configuration parameters for crossover"""

    def __init__(self):
        # Gene inheritance probabilities
        self.inherit_disabled_probability = (
            0.75  # Probability of inheriting disabled gene as disabled
        )
        self.reenable_probability = 0.25  # Probability of re-enabling disabled gene

        # Node inheritance
        self.inherit_random_activation = (
            0.5  # Probability of taking random parent's activation function
        )

        # Bias inheritance
        self.average_bias = True  # Whether to average biases from both parents
        self.bias_mutation_chance = 0.1  # Chance to slightly mutate inherited bias


def crossover_networks(
    parent1: Network,
    parent2: Network,
    fitness1: float,
    fitness2: float,
    config: CrossoverConfig = None,
) -> Network:
    """
    Create offspring by crossing over two parent networks

    Args:
        parent1: First parent network
        parent2: Second parent network
        fitness1: Fitness of first parent
        fitness2: Fitness of second parent
        config: Crossover configuration

    Returns:
        Offspring network
    """
    if config is None:
        config = CrossoverConfig()

    # Determine which parent is more fit
    if fitness1 > fitness2:
        fitter_parent = parent1
        less_fit_parent = parent2
    elif fitness2 > fitness1:
        fitter_parent = parent2
        less_fit_parent = parent1
    else:
        # Equal fitness - randomly choose
        if random.random() < 0.5:
            fitter_parent = parent1
            less_fit_parent = parent2
        else:
            fitter_parent = parent2
            less_fit_parent = parent1

    # Create offspring network with same input/output structure
    offspring = Network(fitter_parent.num_inputs, fitter_parent.num_outputs)

    # Get gene alignment
    gene_alignment = align_genes(parent1, parent2)

    # Inherit nodes first
    inherit_nodes(offspring, fitter_parent, less_fit_parent, gene_alignment, config)

    # Then inherit connections
    inherit_connections(
        offspring, parent1, parent2, gene_alignment, fitness1, fitness2, config
    )

    return offspring


def align_genes(parent1: Network, parent2: Network) -> Dict[str, Dict]:
    """
    Align genes based on innovation numbers

    Args:
        parent1: First parent network
        parent2: Second parent network

    Returns:
        Dictionary containing gene alignment information
    """
    # Get all innovation numbers
    innovations1 = set(parent1.connections.keys())
    innovations2 = set(parent2.connections.keys())

    # Categorize genes
    matching_genes = innovations1.intersection(innovations2)
    excess_genes = set()
    disjoint_genes = set()

    # Calculate excess and disjoint
    if innovations1 and innovations2:
        max_innovation1 = max(innovations1)
        max_innovation2 = max(innovations2)
        min_max = min(max_innovation1, max_innovation2)

        all_innovations = innovations1.union(innovations2)

        for innovation in all_innovations:
            if innovation not in matching_genes:
                if innovation > min_max:
                    excess_genes.add(innovation)
                else:
                    disjoint_genes.add(innovation)

    return {
        "matching": matching_genes,
        "excess": excess_genes,
        "disjoint": disjoint_genes,
        "parent1_only": (innovations1 - matching_genes),
        "parent2_only": (innovations2 - matching_genes),
    }


def inherit_nodes(
    offspring: Network,
    fitter_parent: Network,
    less_fit_parent: Network,
    gene_alignment: Dict,
    config: CrossoverConfig,
) -> Dict[int, int]:
    """
    Inherit nodes from parents, creating mapping from parent node IDs to offspring node IDs

    Args:
        offspring: Offspring network to add nodes to
        fitter_parent: More fit parent
        less_fit_parent: Less fit parent
        gene_alignment: Gene alignment information
        config: Crossover configuration

    Returns:
        Dictionary mapping original node IDs to new node IDs
    """
    node_mapping = {}

    # Input and output nodes already exist in offspring - map them
    for i, input_node in enumerate(fitter_parent.input_nodes):
        node_mapping[input_node.node_id] = offspring.input_nodes[i].node_id

    for i, output_node in enumerate(fitter_parent.output_nodes):
        node_mapping[output_node.node_id] = offspring.output_nodes[i].node_id
        # Inherit bias and activation function
        offspring_node = offspring.output_nodes[i]
        inherit_node_properties(
            offspring_node,
            output_node,
            less_fit_parent.output_nodes[i]
            if i < len(less_fit_parent.output_nodes)
            else None,
            config,
        )

    # Collect all hidden nodes that need to be inherited
    required_hidden_nodes = set()

    # From matching and excess/disjoint genes, find which hidden nodes are needed
    for innovation in gene_alignment["matching"]:
        conn1 = fitter_parent.connections[innovation]
        add_required_hidden_nodes(required_hidden_nodes, conn1)

    for innovation in gene_alignment["parent1_only"]:
        if innovation in fitter_parent.connections:
            conn = fitter_parent.connections[innovation]
            add_required_hidden_nodes(required_hidden_nodes, conn)

    for innovation in gene_alignment["parent2_only"]:
        if innovation in less_fit_parent.connections:
            conn = less_fit_parent.connections[innovation]
            add_required_hidden_nodes(required_hidden_nodes, conn)

    # Create hidden nodes in offspring
    for node_id in required_hidden_nodes:
        # Find the node in either parent
        parent_node = None
        other_parent_node = None

        if node_id in fitter_parent.nodes:
            parent_node = fitter_parent.nodes[node_id]
        if node_id in less_fit_parent.nodes:
            other_parent_node = less_fit_parent.nodes[node_id]
            if parent_node is None:
                parent_node = other_parent_node
                other_parent_node = None

        if parent_node and parent_node.node_type == NodeType.HIDDEN:
            # Create new hidden node
            new_node = Node(
                offspring.next_node_id, NodeType.HIDDEN, parent_node.activation
            )
            offspring.nodes[offspring.next_node_id] = new_node
            offspring.hidden_nodes.append(new_node)

            # Map old ID to new ID
            node_mapping[node_id] = offspring.next_node_id

            # Inherit properties
            inherit_node_properties(new_node, parent_node, other_parent_node, config)

            offspring.next_node_id += 1

    return node_mapping


def add_required_hidden_nodes(required_nodes: set, connection: Connection):
    """Add hidden nodes required by a connection to the set"""
    if connection.input_node.node_type == NodeType.HIDDEN:
        required_nodes.add(connection.input_node.node_id)
    if connection.output_node.node_type == NodeType.HIDDEN:
        required_nodes.add(connection.output_node.node_id)


def inherit_node_properties(
    offspring_node: Node,
    parent1_node: Node,
    parent2_node: Optional[Node],
    config: CrossoverConfig,
):
    """
    Inherit node properties (bias, activation function) from parents

    Args:
        offspring_node: Node in offspring to set properties for
        parent1_node: Node from first parent
        parent2_node: Node from second parent (may be None)
        config: Crossover configuration
    """
    # Inherit activation function
    if parent2_node and random.random() < config.inherit_random_activation:
        offspring_node.activation = random.choice(
            [parent1_node.activation, parent2_node.activation]
        )
    else:
        offspring_node.activation = parent1_node.activation

    # Inherit bias
    if parent2_node and config.average_bias:
        offspring_node.bias = (parent1_node.bias + parent2_node.bias) / 2.0
    else:
        offspring_node.bias = parent1_node.bias

    # Small chance to mutate bias
    if random.random() < config.bias_mutation_chance:
        offspring_node.bias += random.gauss(0, 0.1)


def inherit_connections(
    offspring: Network,
    parent1: Network,
    parent2: Network,
    gene_alignment: Dict,
    fitness1: float,
    fitness2: float,
    config: CrossoverConfig,
):
    """
    Inherit connections from parents

    Args:
        offspring: Offspring network
        parent1: First parent
        parent2: Second parent
        gene_alignment: Gene alignment information
        fitness1: Fitness of parent1
        fitness2: Fitness of parent2
        config: Crossover configuration
    """
    # Determine fitter parent
    fitter_is_parent1 = fitness1 >= fitness2

    # Inherit matching genes
    for innovation in gene_alignment["matching"]:
        conn1 = parent1.connections[innovation]
        conn2 = parent2.connections[innovation]

        inherit_matching_connection(offspring, conn1, conn2, config)

    # Inherit excess and disjoint genes from fitter parent
    fitter_parent = parent1 if fitter_is_parent1 else parent2

    for innovation in (
        gene_alignment["parent1_only"]
        if fitter_is_parent1
        else gene_alignment["parent2_only"]
    ):
        if innovation in fitter_parent.connections:
            conn = fitter_parent.connections[innovation]
            inherit_single_connection(offspring, conn)


def inherit_matching_connection(
    offspring: Network, conn1: Connection, conn2: Connection, config: CrossoverConfig
):
    """
    Inherit a connection gene that exists in both parents

    Args:
        offspring: Offspring network
        conn1: Connection from parent1
        conn2: Connection from parent2
        config: Crossover configuration
    """
    # Find corresponding nodes in offspring
    input_node_id = find_offspring_node_id(offspring, conn1.input_node)
    output_node_id = find_offspring_node_id(offspring, conn1.output_node)

    if input_node_id is None or output_node_id is None:
        return  # Couldn't find corresponding nodes

    # Choose weight (randomly from either parent)
    weight = random.choice([conn1.weight, conn2.weight])

    # Determine if connection should be enabled
    enabled = True
    if not conn1.enabled or not conn2.enabled:
        # If either parent has it disabled
        if not conn1.enabled and not conn2.enabled:
            # Both disabled - inherit as disabled with chance to re-enable
            enabled = random.random() < config.reenable_probability
        else:
            # One disabled - inherit disabled with some probability
            enabled = random.random() >= config.inherit_disabled_probability

    # Create connection in offspring
    innovation_num = innovation_tracker.get_innovation_number(
        input_node_id, output_node_id
    )
    new_connection = Connection(
        innovation_num,
        offspring.nodes[input_node_id],
        offspring.nodes[output_node_id],
        weight,
        enabled,
    )
    offspring.connections[innovation_num] = new_connection


def inherit_single_connection(offspring: Network, conn: Connection):
    """
    Inherit a connection gene from a single parent (excess/disjoint)

    Args:
        offspring: Offspring network
        conn: Connection to inherit
    """
    # Find corresponding nodes in offspring
    input_node_id = find_offspring_node_id(offspring, conn.input_node)
    output_node_id = find_offspring_node_id(offspring, conn.output_node)

    if input_node_id is None or output_node_id is None:
        return  # Couldn't find corresponding nodes

    # Create connection in offspring
    innovation_num = innovation_tracker.get_innovation_number(
        input_node_id, output_node_id
    )
    new_connection = Connection(
        innovation_num,
        offspring.nodes[input_node_id],
        offspring.nodes[output_node_id],
        conn.weight,
        conn.enabled,
    )
    offspring.connections[innovation_num] = new_connection


def find_offspring_node_id(offspring: Network, parent_node: Node) -> Optional[int]:
    """
    Find the corresponding node ID in the offspring for a parent node

    Args:
        offspring: Offspring network
        parent_node: Node from parent

    Returns:
        Node ID in offspring, or None if not found
    """
    if parent_node.node_type == NodeType.INPUT:
        # Find input node by position
        for i, node in enumerate(offspring.input_nodes):
            if i < len(offspring.input_nodes) and node.node_id == i:
                return node.node_id
        # Fallback: find by index
        for i, node in enumerate(offspring.input_nodes):
            if i == parent_node.node_id:
                return node.node_id

    elif parent_node.node_type == NodeType.OUTPUT:
        # Find output node by position
        for i, node in enumerate(offspring.output_nodes):
            if i < len(offspring.output_nodes):
                return node.node_id
        # Fallback: find by relative position
        output_start = len(offspring.input_nodes)
        for i, node in enumerate(offspring.output_nodes):
            if parent_node.node_id == output_start + i:
                return node.node_id

    else:  # HIDDEN
        # For hidden nodes, we need to find by some matching criteria
        # This is simplified - in practice you might need more sophisticated matching
        for node in offspring.hidden_nodes:
            # Simple heuristic: if only one hidden node exists, use it
            if len(offspring.hidden_nodes) == 1:
                return node.node_id

        # More sophisticated approach: find by innovation number patterns
        # This is a simplification - real implementation might track node innovations
        for node in offspring.hidden_nodes:
            return node.node_id  # Return first match for now

    return None


if __name__ == "__main__":
    # Test crossover
    print("Testing NEAT Crossover...")

    # Set seed for reproducible results
    random.seed(42)

    # Create two parent networks
    parent1 = Network(num_inputs=3, num_outputs=2)
    parent2 = Network(num_inputs=3, num_outputs=2)

    # Add some connections to parent1
    parent1.add_connection(0, 3, weight=0.5)  # input 0 -> output 0
    parent1.add_connection(1, 4, weight=-0.3)  # input 1 -> output 1
    parent1.add_connection(2, 3, weight=0.8)  # input 2 -> output 0

    # Add some different connections to parent2
    parent2.add_connection(0, 3, weight=0.7)  # Same connection, different weight
    parent2.add_connection(1, 3, weight=0.2)  # Different connection
    parent2.add_connection(2, 4, weight=-0.5)  # Different connection

    # Add hidden nodes to make it more interesting
    if parent1.connections:
        first_conn = list(parent1.connections.values())[0]
        parent1.add_node(first_conn)

    if parent2.connections:
        first_conn = list(parent2.connections.values())[0]
        parent2.add_node(first_conn)

    print(f"Parent 1 complexity: {parent1.get_complexity()}")
    print(f"Parent 2 complexity: {parent2.get_complexity()}")

    # Assign fitness values
    fitness1 = 85.0
    fitness2 = 70.0

    # Perform crossover
    config = CrossoverConfig()
    offspring = crossover_networks(parent1, parent2, fitness1, fitness2, config)

    print(f"Offspring complexity: {offspring.get_complexity()}")

    # Test forward pass to ensure offspring works
    test_inputs = [0.5, -0.2, 0.8]
    try:
        outputs = offspring.forward_pass(test_inputs)
        print(f"Offspring forward pass: {test_inputs} -> {outputs}")
        print("Crossover successful!")
    except Exception as e:
        print(f"Offspring forward pass failed: {e}")

    # Test multiple crossovers
    print("\nTesting multiple crossovers...")
    for i in range(5):
        child = crossover_networks(parent1, parent2, fitness1, fitness2, config)
        complexity = child.get_complexity()
        print(f"Child {i + 1}: {complexity}")

        # Quick functionality test
        try:
            outputs = child.forward_pass(test_inputs)
            print("  Forward pass: ✓")
        except Exception as e:
            print("  Forward pass: ✗ ", e)

    print("Crossover testing complete!")

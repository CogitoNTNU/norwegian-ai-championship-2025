import random
import math
import time
from typing import List, Dict

# Try to import optional packages
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from network import Network, ActivationFunction, innovation_tracker
from mutations import MutationConfig, mutate_network
from speciation import SpeciationManager, SpeciationConfig
from crossover import CrossoverConfig, crossover_networks


class SimpleNEATConfig:
    """Simplified NEAT configuration"""

    def __init__(self):
        # Population
        self.population_size = 100
        self.num_inputs = 1
        self.num_outputs = 1

        # Evolution
        self.max_generations = 500
        self.fitness_threshold = 0.95

        # NO SURVIVORS - complete generation replacement
        self.survival_rate = 0.0
        self.crossover_rate = 0.75

        # Mutation rates
        self.add_connection_rate = 0.5
        self.add_node_rate = 0.2
        self.weight_mutation_rate = 0.8
        self.bias_mutation_rate = 0.3

        # Speciation
        self.compatibility_threshold = 3.0
        self.target_species = 5


class SineFitnessEvaluator:
    """Evaluates networks on sine function"""

    def __init__(self):
        # Test points for sine function
        self.test_points = 20
        self.x_values = []
        self.y_targets = []

        for i in range(self.test_points):
            x = -math.pi + (2 * math.pi * i / (self.test_points - 1))
            self.x_values.append(x)
            self.y_targets.append(math.sin(x))

    def evaluate_network(self, network: Network) -> float:
        """Evaluate network fitness: 1 / (1 + MSE)"""
        total_error = 0.0

        for i in range(self.test_points):
            try:
                outputs = network.forward_pass([self.x_values[i]])
                if not outputs:
                    return 0.001

                prediction = outputs[0]
                if math.isnan(prediction) or math.isinf(prediction):
                    return 0.001

                error = (prediction - self.y_targets[i]) ** 2
                total_error += error

            except Exception:
                return 0.001

        mse = total_error / self.test_points
        fitness = 1.0 / (1.0 + mse)
        return max(0.001, min(1.0, fitness))


class SimpleNEATEvolution:
    """Clean, simple NEAT implementation"""

    def __init__(self, config: SimpleNEATConfig = None, seed: int = None):
        self.config = config or SimpleNEATConfig()

        if seed is not None:
            random.seed(seed)
            if HAS_NUMPY:
                np.random.seed(seed)

        # Initialize components
        self.fitness_evaluator = SineFitnessEvaluator()

        # Setup mutation config
        self.mutation_config = MutationConfig()
        self.mutation_config.add_connection_rate = self.config.add_connection_rate
        self.mutation_config.add_node_rate = self.config.add_node_rate
        self.mutation_config.weight_mutation_rate = self.config.weight_mutation_rate
        self.mutation_config.bias_mutation_rate = self.config.bias_mutation_rate
        self.mutation_config.weight_perturbation_strength = 0.5
        self.mutation_config.weight_replacement_range = 3.0

        # Setup speciation
        spec_config = SpeciationConfig()
        spec_config.compatibility_threshold = self.config.compatibility_threshold
        spec_config.target_species = self.config.target_species
        self.speciation_manager = SpeciationManager(spec_config)

        # Setup crossover
        self.crossover_config = CrossoverConfig()

        # Evolution tracking
        self.generation = 0
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.species_count_history = []

        innovation_tracker.reset()

    def create_initial_population(self) -> List[Network]:
        """Create initial population with basic networks"""
        population = []

        for _ in range(self.config.population_size):
            network = Network(self.config.num_inputs, self.config.num_outputs)

            # Set LINEAR activation for output (critical for sine function)
            for output_node in network.output_nodes:
                output_node.activation = ActivationFunction.LINEAR
                output_node.bias = random.uniform(-1.0, 1.0)

            # Add initial connection
            network.add_connection(0, 1, weight=random.uniform(-2.0, 2.0))
            population.append(network)

        return population

    def evaluate_population(self, population: List[Network]) -> List[float]:
        """Evaluate entire population"""
        fitness_scores = []

        for network in population:
            fitness = self.fitness_evaluator.evaluate_network(network)
            network.fitness = fitness
            fitness_scores.append(fitness)

        return fitness_scores

    def select_parents(
        self, species_networks: List[Network], num_offspring: int
    ) -> List[Network]:
        """Select parents from a species using tournament selection"""
        if not species_networks:
            return []

        parents = []
        tournament_size = min(3, len(species_networks))

        for _ in range(num_offspring):
            # Tournament selection
            tournament = random.sample(species_networks, tournament_size)
            winner = max(tournament, key=lambda x: getattr(x, "fitness", 0))
            parents.append(winner)

        return parents

    def copy_network(self, original: Network) -> Network:
        """Create a deep copy of a network"""
        copy = Network(original.num_inputs, original.num_outputs)

        # Copy output node properties
        for i, orig_output in enumerate(original.output_nodes):
            copy.output_nodes[i].bias = orig_output.bias
            copy.output_nodes[i].activation = orig_output.activation

        # Copy hidden nodes
        node_mapping = {}
        for i in range(len(original.input_nodes)):
            node_mapping[original.input_nodes[i].node_id] = copy.input_nodes[i].node_id
        for i in range(len(original.output_nodes)):
            node_mapping[original.output_nodes[i].node_id] = copy.output_nodes[
                i
            ].node_id

        # Add hidden nodes
        for hidden_node in original.hidden_nodes:
            from network import Node, NodeType

            new_node = Node(copy.next_node_id, NodeType.HIDDEN, hidden_node.activation)
            new_node.bias = hidden_node.bias
            copy.nodes[copy.next_node_id] = new_node
            copy.hidden_nodes.append(new_node)
            node_mapping[hidden_node.node_id] = copy.next_node_id
            copy.next_node_id += 1

        # Copy connections
        for conn in original.connections.values():
            if (
                conn.input_node.node_id in node_mapping
                and conn.output_node.node_id in node_mapping
            ):
                new_input_id = node_mapping[conn.input_node.node_id]
                new_output_id = node_mapping[conn.output_node.node_id]
                new_conn = copy.add_connection(new_input_id, new_output_id, conn.weight)
                if new_conn:
                    new_conn.enabled = conn.enabled

        return copy

    def create_offspring(
        self, species_networks: List[Network], num_offspring: int
    ) -> List[Network]:
        """Create offspring for a species"""
        if num_offspring <= 0:
            return []

        offspring = []
        parents = self.select_parents(
            species_networks, num_offspring * 2
        )  # Extra parents for crossover

        for i in range(num_offspring):
            if len(parents) >= 2 and random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                # Ensure different parents
                attempts = 0
                while parent1 == parent2 and len(parents) > 1 and attempts < 5:
                    parent2 = random.choice(parents)
                    attempts += 1

                fitness1 = getattr(parent1, "fitness", 0.0)
                fitness2 = getattr(parent2, "fitness", 0.0)

                child = crossover_networks(
                    parent1, parent2, fitness1, fitness2, self.crossover_config
                )
            else:
                # Asexual reproduction
                parent = random.choice(parents) if parents else species_networks[0]
                child = self.copy_network(parent)

            # Mutate child
            mutate_network(child, self.mutation_config)
            offspring.append(child)

        return offspring

    def evolve(self) -> Dict:
        """Main evolution loop"""
        print("Starting Simple NEAT Evolution")
        print(
            f"Population: {self.config.population_size}, Generations: {self.config.max_generations}"
        )
        print(f"Target fitness: {self.config.fitness_threshold}")
        print("=" * 60)

        # Create initial population
        self.population = self.create_initial_population()
        start_time = time.time()

        for generation in range(self.config.max_generations):
            gen_start = time.time()
            self.generation = generation

            # Evaluate fitness
            fitness_scores = self.evaluate_population(self.population)

            # Calculate statistics
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Assign to species
            self.speciation_manager.assign_to_species(self.population)
            species_count = len(self.speciation_manager.species)
            self.species_count_history.append(species_count)

            gen_time = time.time() - gen_start

            # Print progress
            if generation % 10 == 0 or best_fitness >= self.config.fitness_threshold:
                print(
                    f"Gen {generation:3d}: Best={best_fitness:.4f}, "
                    f"Avg={avg_fitness:.4f}, Species={species_count}, "
                    f"Pop={len(self.population)}, Time={gen_time:.3f}s"
                )

            # Check if target reached
            if best_fitness >= self.config.fitness_threshold:
                print(f"Target fitness reached at generation {generation}!")
                break

            # Create next generation (NO SURVIVORS)
            if generation < self.config.max_generations - 1:
                new_population = []

                # Calculate offspring per species
                total_fitness = sum(
                    s.calculate_adjusted_fitness()
                    for s in self.speciation_manager.species
                )

                if total_fitness > 0:
                    for species in self.speciation_manager.species:
                        species_fitness = species.calculate_adjusted_fitness()
                        species_ratio = species_fitness / total_fitness
                        offspring_count = max(
                            1, int(species_ratio * self.config.population_size)
                        )

                        offspring = self.create_offspring(
                            species.members, offspring_count
                        )
                        new_population.extend(offspring)
                else:
                    # Fallback: equal distribution
                    offspring_per_species = max(
                        1,
                        self.config.population_size
                        // len(self.speciation_manager.species),
                    )
                    for species in self.speciation_manager.species:
                        offspring = self.create_offspring(
                            species.members, offspring_per_species
                        )
                        new_population.extend(offspring)

                # Ensure exact population size
                if len(new_population) > self.config.population_size:
                    new_population = random.sample(
                        new_population, self.config.population_size
                    )
                elif len(new_population) < self.config.population_size:
                    # Add random networks
                    needed = self.config.population_size - len(new_population)
                    for _ in range(needed):
                        network = Network(
                            self.config.num_inputs, self.config.num_outputs
                        )
                        for output_node in network.output_nodes:
                            output_node.activation = ActivationFunction.LINEAR
                            output_node.bias = random.uniform(-1.0, 1.0)
                        network.add_connection(0, 1, weight=random.uniform(-2.0, 2.0))
                        new_population.append(network)

                self.population = new_population

        # Final evaluation
        final_fitness = self.evaluate_population(self.population)
        best_idx = final_fitness.index(max(final_fitness))
        best_network = self.population[best_idx]

        end_time = time.time()
        print("=" * 60)
        print(f"Evolution completed in {end_time - start_time:.2f} seconds")
        print(f"Final best fitness: {max(final_fitness):.4f}")
        print(f"Best network complexity: {best_network.get_complexity()}")

        return {
            "best_network": best_network,
            "best_fitness": max(final_fitness),
            "generation": self.generation,
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
            "species_count_history": self.species_count_history,
        }

    def test_best_network(self, network: Network):
        """Test the best network on sine function"""
        print("\nTesting best network:")
        test_points = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]

        for x in test_points:
            try:
                prediction = network.forward_pass([x])[0]
                actual = math.sin(x)
                error = abs(prediction - actual)
                print(
                    f"sin({x:.3f}) = {actual:.3f}, predicted = {prediction:.3f}, error = {error:.3f}"
                )
            except Exception as e:
                print(f"Network failed on input {x}: {e}")

    def plot_results(self, results: Dict):
        """Plot evolution results"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        generations = range(len(self.best_fitness_history))

        # Fitness evolution
        ax1.plot(generations, self.best_fitness_history, "b-", label="Best")
        ax1.plot(generations, self.avg_fitness_history, "r-", label="Average")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title("Fitness Evolution")
        ax1.legend()
        ax1.grid(True)

        # Species count
        ax2.plot(generations, self.species_count_history, "g-")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Species Count")
        ax2.set_title("Species Diversity")
        ax2.grid(True)

        # Network performance
        best_network = results["best_network"]
        if HAS_NUMPY:
            test_x = np.linspace(-math.pi, math.pi, 100)
            predictions = []
            for x in test_x:
                try:
                    pred = best_network.forward_pass([x])[0]
                    predictions.append(pred)
                except Exception:
                    predictions.append(0.0)

            ax3.plot(test_x, np.sin(test_x), "b-", label="True Sine", linewidth=2)
            ax3.plot(test_x, predictions, "r--", label="Network", linewidth=2)
        else:
            test_x = [i * 0.1 - math.pi for i in range(int(2 * math.pi / 0.1))]
            true_y = [math.sin(x) for x in test_x]
            pred_y = []
            for x in test_x:
                try:
                    pred = best_network.forward_pass([x])[0]
                    pred_y.append(pred)
                except Exception as e:
                    pred_y.append(0.0)
                    print(f"Error occurred while processing {x}: {e}")

            ax3.plot(test_x, true_y, "b-", label="True Sine", linewidth=2)
            ax3.plot(test_x, pred_y, "r--", label="Network", linewidth=2)

        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_title("Sine Approximation")
        ax3.legend()
        ax3.grid(True)

        # Network structure info
        complexity = best_network.get_complexity()
        ax4.text(
            0.1,
            0.7,
            f"Nodes: {complexity['nodes']}",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.text(
            0.1,
            0.6,
            f"Connections: {complexity['enabled_connections']}",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.text(
            0.1,
            0.5,
            f"Final Fitness: {results['best_fitness']:.4f}",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.text(
            0.1,
            0.4,
            f"Generation: {results['generation']}",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.set_title("Best Network Info")
        ax4.axis("off")

        plt.tight_layout()
        plt.show()


def main():
    """Main function"""
    print("Simple NEAT Evolution for Sine Function")
    print("=" * 40)

    config = SimpleNEATConfig()
    evolution = SimpleNEATEvolution(config, seed=42)

    # Run evolution
    results = evolution.evolve()

    # Test best network
    evolution.test_best_network(results["best_network"])

    # Plot if available
    try:
        evolution.plot_results(results)
    except Exception as e:
        print(f"Plotting failed: {e}")

    return results


if __name__ == "__main__":
    results = main()

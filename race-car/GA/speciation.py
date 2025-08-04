import random
from typing import List, Dict, Optional
from network import Network


class SpeciationConfig:
    """Configuration parameters for speciation"""

    def __init__(self):
        # Simple distance calculation
        self.excess_coefficient = 1.0
        self.disjoint_coefficient = 1.0
        self.weight_difference_coefficient = 0.4

        # Compatibility threshold - much more permissive
        self.compatibility_threshold = 3.0

        # Dynamic threshold adjustment
        self.target_species = 5  # Much fewer target species
        self.threshold_modifier = 0.5  # Bigger adjustments

        # Species survival
        self.survival_threshold = 0.2
        self.elite_size = 1

        # Stagnation parameters
        self.max_stagnation = 15
        self.species_elitism_threshold = 5


class Species:
    """Represents a species of similar networks"""

    def __init__(self, species_id: int, representative: Network):
        self.species_id = species_id
        self.representative = representative  # Representative genome for this species
        self.members: List[Network] = []
        self.fitness_history: List[float] = []
        self.best_fitness = float("-inf")
        self.generations_without_improvement = 0
        self.age = 0

    def add_member(self, network: Network):
        """Add a network to this species"""
        self.members.append(network)

    def clear_members(self):
        """Clear all members (but keep representative)"""
        self.members.clear()

    def update_representative(self):
        """Update the representative to a random member"""
        if self.members:
            import random

            self.representative = random.choice(self.members)

    def calculate_average_fitness(self) -> float:
        """Calculate the average fitness of all members"""
        if not self.members:
            return 0.0
        return sum(getattr(member, "fitness", 0.0) for member in self.members) / len(
            self.members
        )

    def calculate_adjusted_fitness(self) -> float:
        """Calculate fitness adjusted for species size (fitness sharing)"""
        if not self.members:
            return 0.0
        total_fitness = sum(getattr(member, "fitness", 0.0) for member in self.members)
        return total_fitness / len(self.members)

    def get_best_member(self) -> Optional[Network]:
        """Get the member with the highest fitness"""
        if not self.members:
            return None
        return max(self.members, key=lambda x: getattr(x, "fitness", float("-inf")))

    def update_stagnation(self, current_best_fitness: float):
        """Update stagnation counter based on fitness improvement"""
        self.age += 1

        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        # Add to fitness history
        self.fitness_history.append(current_best_fitness)

        # Keep only recent history
        if len(self.fitness_history) > 20:
            self.fitness_history.pop(0)

    def is_stagnant(self, max_stagnation: int) -> bool:
        """Check if this species is stagnant"""
        return self.generations_without_improvement >= max_stagnation

    def __str__(self):
        return f"Species(id={self.species_id}, members={len(self.members)}, age={self.age})"


class SpeciationManager:
    """Manages species and compatibility calculations"""

    def __init__(self, config: SpeciationConfig = None):
        self.config = config or SpeciationConfig()
        self.species: List[Species] = []
        self.species_counter = 0
        self.generation = 0

    def calculate_distance(self, network1: Network, network2: Network) -> float:
        """
        Calculate the compatibility distance between two networks
        Simple and robust approach
        """
        # Get connection innovation numbers
        innovations1 = set(network1.connections.keys())
        innovations2 = set(network2.connections.keys())

        # Handle empty connections
        if len(innovations1) == 0 and len(innovations2) == 0:
            return 0.0
        if len(innovations1) == 0 or len(innovations2) == 0:
            return max(len(innovations1), len(innovations2))

        # Find matching and non-matching genes
        matching = innovations1.intersection(innovations2)
        all_innovations = innovations1.union(innovations2)
        non_matching = len(all_innovations) - len(matching)

        # Calculate weight differences for matching connections
        weight_diff = 0.0
        if matching:
            for innovation in matching:
                conn1 = network1.connections[innovation]
                conn2 = network2.connections[innovation]
                weight_diff += abs(conn1.weight - conn2.weight)
            weight_diff /= len(matching)

        # Simple distance calculation
        # Count total structural differences plus weight differences
        structural_diff = non_matching

        # Normalize by larger genome size (minimum 1 to avoid division by zero)
        N = max(len(innovations1), len(innovations2), 1)

        # Use N to normalize structural differences for fair comparison
        distance = (structural_diff / N) + (
            self.config.weight_difference_coefficient * weight_diff
        )

        return distance

    def assign_to_species(self, networks: List[Network]):
        """
        Assign networks to species based on compatibility

        Args:
            networks: List of networks to assign
        """
        # Clear existing members but keep species structure
        for species in self.species:
            species.clear_members()

        # Assign each network to a species
        unassigned_networks = []

        for network in networks:
            assigned = False

            # Try to assign to existing species
            for species in self.species:
                distance = self.calculate_distance(network, species.representative)
                if distance < self.config.compatibility_threshold:
                    species.add_member(network)
                    assigned = True
                    break

            if not assigned:
                unassigned_networks.append(network)

        # Create new species for unassigned networks
        for network in unassigned_networks:
            new_species = Species(self.species_counter, network)
            new_species.add_member(network)
            self.species.append(new_species)
            self.species_counter += 1

        # Remove empty species
        self.species = [s for s in self.species if s.members]

        # Update representatives
        for species in self.species:
            species.update_representative()

    def update_species_fitness(self):
        """Update fitness information for all species"""
        for species in self.species:
            if species.members:
                best_member = species.get_best_member()
                if best_member:
                    best_fitness = getattr(best_member, "fitness", float("-inf"))
                    species.update_stagnation(best_fitness)

    def remove_stagnant_species(self):
        """Remove species that have been stagnant for too long"""
        # Always keep at least one species
        if len(self.species) <= 1:
            return

        non_stagnant = []
        for species in self.species:
            if not species.is_stagnant(self.config.max_stagnation):
                non_stagnant.append(species)

        # Keep at least the best species even if stagnant
        if not non_stagnant:
            best_species = max(self.species, key=lambda s: s.best_fitness)
            non_stagnant.append(best_species)

        self.species = non_stagnant

    def adjust_compatibility_threshold(self):
        """Dynamically adjust compatibility threshold to maintain target species count"""
        current_species_count = len(self.species)
        target = self.config.target_species

        if current_species_count > target:
            # Too many species, increase threshold (merge species)
            self.config.compatibility_threshold += self.config.threshold_modifier
        elif current_species_count < target:
            # Too few species, decrease threshold (split species)
            self.config.compatibility_threshold -= self.config.threshold_modifier

        # Keep threshold in reasonable bounds
        self.config.compatibility_threshold = max(
            0.1, self.config.compatibility_threshold
        )

    def calculate_species_sizes(self, total_population: int) -> Dict[int, int]:
        """
        Calculate how many offspring each species should produce

        Args:
            total_population: Total number of offspring to produce

        Returns:
            Dictionary mapping species_id to number of offspring
        """
        species_sizes = {}

        if not self.species:
            return species_sizes

        # Calculate adjusted fitness for each species
        total_adjusted_fitness = 0.0
        species_fitness = {}

        for species in self.species:
            adjusted_fitness = species.calculate_adjusted_fitness()
            species_fitness[species.species_id] = adjusted_fitness
            total_adjusted_fitness += adjusted_fitness

        # Allocate offspring based on relative fitness
        if total_adjusted_fitness > 0:
            for species in self.species:
                relative_fitness = (
                    species_fitness[species.species_id] / total_adjusted_fitness
                )
                offspring_count = int(relative_fitness * total_population)
                species_sizes[species.species_id] = max(
                    1, offspring_count
                )  # At least 1 offspring
        else:
            # Equal allocation if no fitness information
            offspring_per_species = max(1, total_population // len(self.species))
            for species in self.species:
                species_sizes[species.species_id] = offspring_per_species

        # Ensure total doesn't exceed population size
        total_allocated = sum(species_sizes.values())
        if total_allocated > total_population:
            # Reduce largest allocations
            sorted_species = sorted(
                species_sizes.items(), key=lambda x: x[1], reverse=True
            )
            excess = total_allocated - total_population
            for species_id, size in sorted_species:
                if excess <= 0:
                    break
                reduction = min(excess, size - 1)  # Keep at least 1
                species_sizes[species_id] -= reduction
                excess -= reduction

        return species_sizes

    def get_species_statistics(self) -> Dict:
        """Get statistics about current species"""
        if not self.species:
            return {"species_count": 0, "total_members": 0}

        stats = {
            "species_count": len(self.species),
            "total_members": sum(len(s.members) for s in self.species),
            "average_species_size": sum(len(s.members) for s in self.species)
            / len(self.species),
            "largest_species": max(len(s.members) for s in self.species),
            "smallest_species": min(len(s.members) for s in self.species),
            "compatibility_threshold": self.config.compatibility_threshold,
            "stagnant_species": sum(
                1 for s in self.species if s.is_stagnant(self.config.max_stagnation)
            ),
        }

        return stats

    def advance_generation(self):
        """Advance to the next generation"""
        self.generation += 1
        self.update_species_fitness()
        self.remove_stagnant_species()
        self.adjust_compatibility_threshold()


if __name__ == "__main__":
    # Test speciation
    print("Testing NEAT Speciation...")

    # Create some test networks
    networks = []
    for i in range(50):
        network = Network(num_inputs=3, num_outputs=2)

        # Add some random connections to create diversity
        import random

        random.seed(i)  # Different seed for each network

        # Add random connections
        for _ in range(random.randint(1, 5)):
            input_nodes = [n.node_id for n in network.input_nodes]
            output_nodes = [n.node_id for n in network.output_nodes]

            if random.random() < 0.7:  # 70% chance to add connection
                input_id = random.choice(input_nodes)
                output_id = random.choice(output_nodes)
                network.add_connection(input_id, output_id)

        # Assign random fitness
        network.fitness = random.uniform(0, 100)
        networks.append(network)

    # Test speciation
    speciation_manager = SpeciationManager()

    print(f"Created {len(networks)} networks")

    # Assign to species
    speciation_manager.assign_to_species(networks)

    # Print statistics
    stats = speciation_manager.get_species_statistics()
    print(f"Speciation results: {stats}")

    # Print species details
    for species in speciation_manager.species:
        print(f"{species} - Best fitness: {species.best_fitness:.2f}")

    # Test distance calculation
    if len(networks) >= 2:
        distance = speciation_manager.calculate_distance(networks[0], networks[1])
        print(f"Distance between first two networks: {distance:.3f}")

    # Test multiple generations
    print("\nTesting multiple generations...")
    for gen in range(5):
        speciation_manager.advance_generation()
        stats = speciation_manager.get_species_statistics()
        print(
            f"Generation {gen + 1}: {stats['species_count']} species, threshold: {stats['compatibility_threshold']:.2f}"
        )

    print("Speciation testing complete!")

import random
from typing import List, Dict, Tuple, Optional
from network import Network
from speciation import Species, SpeciationManager


class SelectionConfig:
    """Configuration parameters for selection"""

    def __init__(self):
        # Selection methods
        self.selection_method = "tournament"  # "tournament", "roulette", "rank"
        self.tournament_size = 3

        # Survival rates
        self.survival_rate = 0.2  # Fraction of population that survives
        self.elite_count = 1  # Number of best individuals that always survive

        # Reproduction rates
        self.crossover_rate = 0.75  # Probability of crossover vs asexual reproduction
        self.interspecies_mating_rate = 0.001  # Rate of mating between species

        # Species protection
        self.min_species_size = 1  # Minimum individuals per species
        self.species_elitism = True  # Whether to always keep best from each species


def select_parents(
    species: Species, num_parents: int, config: SelectionConfig
) -> List[Network]:
    """
    Select parent networks from a species for reproduction

    Args:
        species: Species to select from
        num_parents: Number of parents to select
        config: Selection configuration

    Returns:
        List of selected parent networks
    """
    if not species.members:
        return []

    if num_parents <= 0:
        return []

    # Ensure we don't select more parents than available
    num_parents = min(num_parents, len(species.members))

    if config.selection_method == "tournament":
        return tournament_selection(
            species.members, num_parents, config.tournament_size
        )
    elif config.selection_method == "roulette":
        return roulette_wheel_selection(species.members, num_parents)
    elif config.selection_method == "rank":
        return rank_selection(species.members, num_parents)
    else:
        # Default to tournament
        return tournament_selection(
            species.members, num_parents, config.tournament_size
        )


def tournament_selection(
    population: List[Network], num_parents: int, tournament_size: int
) -> List[Network]:
    """
    Select parents using tournament selection

    Args:
        population: Population to select from
        num_parents: Number of parents to select
        tournament_size: Size of each tournament

    Returns:
        List of selected parents
    """
    selected = []

    for _ in range(num_parents):
        # Run tournament
        tournament = random.sample(population, min(tournament_size, len(population)))
        winner = max(tournament, key=lambda x: getattr(x, "fitness", 0.0))
        selected.append(winner)

    return selected


def roulette_wheel_selection(
    population: List[Network], num_parents: int
) -> List[Network]:
    """
    Select parents using roulette wheel selection (fitness proportionate)

    Args:
        population: Population to select from
        num_parents: Number of parents to select

    Returns:
        List of selected parents
    """
    if not population:
        return []

    # Calculate fitness values and handle negative fitness
    fitness_values = [getattr(network, "fitness", 0.0) for network in population]
    min_fitness = min(fitness_values)

    # Shift fitness values to be non-negative
    if min_fitness < 0:
        fitness_values = [f - min_fitness + 1 for f in fitness_values]

    total_fitness = sum(fitness_values)

    selected = []

    for _ in range(num_parents):
        if total_fitness <= 0:
            # If no positive fitness, select randomly
            selected.append(random.choice(population))
        else:
            # Spin the roulette wheel
            pick = random.uniform(0, total_fitness)
            current = 0

            for i, fitness in enumerate(fitness_values):
                current += fitness
                if current >= pick:
                    selected.append(population[i])
                    break
            else:
                # Fallback if rounding errors
                selected.append(population[-1])

    return selected


def rank_selection(population: List[Network], num_parents: int) -> List[Network]:
    """
    Select parents using rank-based selection

    Args:
        population: Population to select from
        num_parents: Number of parents to select

    Returns:
        List of selected parents
    """
    if not population:
        return []

    # Sort population by fitness
    sorted_population = sorted(population, key=lambda x: getattr(x, "fitness", 0.0))

    # Assign ranks (1 = worst, n = best)
    ranks = list(range(1, len(sorted_population) + 1))
    total_rank = sum(ranks)

    selected = []

    for _ in range(num_parents):
        # Select based on rank
        pick = random.uniform(0, total_rank)
        current = 0

        for i, rank in enumerate(ranks):
            current += rank
            if current >= pick:
                selected.append(sorted_population[i])
                break
        else:
            # Fallback
            selected.append(sorted_population[-1])

    return selected


def select_survivors(
    species: Species, survival_count: int, config: SelectionConfig
) -> List[Network]:
    """
    Select which individuals survive to the next generation

    Args:
        species: Species to select survivors from
        survival_count: Number of survivors to select
        config: Selection configuration

    Returns:
        List of surviving networks
    """
    if not species.members or survival_count <= 0:
        return []

    # Sort by fitness (descending)
    sorted_members = sorted(
        species.members,
        key=lambda x: getattr(x, "fitness", float("-inf")),
        reverse=True,
    )

    # Always keep the elite
    elite_count = min(config.elite_count, len(sorted_members), survival_count)
    survivors = sorted_members[:elite_count]

    # Select remaining survivors
    remaining_slots = survival_count - elite_count
    if remaining_slots > 0 and len(sorted_members) > elite_count:
        remaining_candidates = sorted_members[elite_count:]

        if config.selection_method == "tournament":
            additional = tournament_selection(
                remaining_candidates, remaining_slots, config.tournament_size
            )
        elif config.selection_method == "roulette":
            additional = roulette_wheel_selection(remaining_candidates, remaining_slots)
        elif config.selection_method == "rank":
            additional = rank_selection(remaining_candidates, remaining_slots)
        else:
            # Default: take top performers
            additional = remaining_candidates[:remaining_slots]

        survivors.extend(additional)

    return survivors


def calculate_reproduction_quotas(
    speciation_manager: SpeciationManager, population_size: int, config: SelectionConfig
) -> Dict[int, Dict[str, int]]:
    """
    Calculate how many offspring each species should produce and how many survive

    Args:
        speciation_manager: Manager containing all species
        population_size: Target population size
        config: Selection configuration

    Returns:
        Dictionary with reproduction quotas for each species
    """
    quotas = {}

    if not speciation_manager.species:
        return quotas

    # Calculate total adjusted fitness across all species
    total_adjusted_fitness = 0.0
    species_fitness = {}

    for species in speciation_manager.species:
        # Calculate species fitness (sum of all member fitness / species size)
        if species.members:
            total_species_fitness = sum(
                getattr(member, "fitness", 0.0) for member in species.members
            )
            adjusted_fitness = total_species_fitness / len(
                species.members
            )  # Fitness sharing
            species_fitness[species.species_id] = adjusted_fitness
            total_adjusted_fitness += adjusted_fitness
        else:
            species_fitness[species.species_id] = 0.0

    # Allocate offspring based on relative fitness
    total_offspring_allocated = 0

    for species in speciation_manager.species:
        species_id = species.species_id

        if total_adjusted_fitness > 0:
            # Proportional allocation based on fitness
            relative_fitness = species_fitness[species_id] / total_adjusted_fitness
            offspring_count = int(relative_fitness * population_size)
        else:
            # Equal allocation if no fitness information
            offspring_count = population_size // len(speciation_manager.species)

        # Ensure minimum species size
        offspring_count = max(config.min_species_size, offspring_count)

        # Calculate survivors (some individuals carry over to next generation)
        survival_count = max(1, int(len(species.members) * config.survival_rate))
        if config.species_elitism:
            survival_count = max(survival_count, 1)  # Always keep at least the best

        quotas[species_id] = {
            "offspring": offspring_count,
            "survivors": min(survival_count, len(species.members)),
            "new_individuals": max(0, offspring_count - survival_count),
        }

        total_offspring_allocated += offspring_count

    # Adjust if we've allocated too many or too few
    difference = population_size - total_offspring_allocated

    if difference != 0:
        # Distribute the difference proportionally
        sorted_species = sorted(
            quotas.items(), key=lambda x: x[1]["offspring"], reverse=True
        )

        for species_id, quota in sorted_species:
            if difference == 0:
                break

            if difference > 0:
                # Add offspring to largest species
                quotas[species_id]["offspring"] += 1
                quotas[species_id]["new_individuals"] += 1
                difference -= 1
            else:
                # Remove offspring from largest species (but maintain minimums)
                if quota["offspring"] > config.min_species_size:
                    quotas[species_id]["offspring"] -= 1
                    if quota["new_individuals"] > 0:
                        quotas[species_id]["new_individuals"] -= 1
                    else:
                        quotas[species_id]["survivors"] = max(0, quota["survivors"] - 1)
                    difference += 1

    return quotas


def select_for_crossover(
    species: Species, config: SelectionConfig
) -> Tuple[Optional[Network], Optional[Network]]:
    """
    Select two parents for crossover reproduction

    Args:
        species: Species to select from
        config: Selection configuration

    Returns:
        Tuple of (parent1, parent2) or (None, None) if not possible
    """
    if len(species.members) < 2:
        return None, None

    # Select two different parents
    parents = select_parents(species, 2, config)

    if len(parents) >= 2:
        return parents[0], parents[1]
    elif len(parents) == 1:
        # If we only got one parent, try to get another different one
        other_candidates = [m for m in species.members if m != parents[0]]
        if other_candidates:
            return parents[0], random.choice(other_candidates)

    return None, None


def select_for_interspecies_mating(
    speciation_manager: SpeciationManager,
    current_species: Species,
    config: SelectionConfig,
) -> Optional[Network]:
    """
    Select a parent from a different species for interspecies mating

    Args:
        speciation_manager: Manager containing all species
        current_species: The species we're selecting a mate for
        config: Selection configuration

    Returns:
        Network from different species or None
    """
    # Get all other species
    other_species = [
        s
        for s in speciation_manager.species
        if s.species_id != current_species.species_id
    ]

    if not other_species:
        return None

    # Select a random species
    mate_species = random.choice(other_species)

    if not mate_species.members:
        return None

    # Select a parent from that species
    parents = select_parents(mate_species, 1, config)
    return parents[0] if parents else None


class SelectionManager:
    """Manages the selection process for NEAT evolution"""

    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()

    def select_next_generation(
        self, speciation_manager: SpeciationManager, population_size: int
    ) -> Dict[int, Dict[str, List[Network]]]:
        """
        Select survivors and reproduction quotas for the next generation

        Args:
            speciation_manager: Manager containing all species
            population_size: Target population size

        Returns:
            Dictionary with selection results for each species
        """
        # Calculate reproduction quotas
        quotas = calculate_reproduction_quotas(
            speciation_manager, population_size, self.config
        )

        results = {}

        for species in speciation_manager.species:
            species_id = species.species_id
            quota = quotas.get(
                species_id, {"survivors": 0, "offspring": 0, "new_individuals": 0}
            )

            # Select survivors
            survivors = select_survivors(species, quota["survivors"], self.config)

            # Prepare for reproduction
            reproduction_parents = []
            if quota["new_individuals"] > 0:
                # Select parents for reproduction
                parent_count = min(
                    len(species.members), max(2, quota["new_individuals"])
                )
                reproduction_parents = select_parents(
                    species, parent_count, self.config
                )

            results[species_id] = {
                "survivors": survivors,
                "reproduction_parents": reproduction_parents,
                "quota": quota,
            }

        return results

    def get_selection_statistics(self, results: Dict) -> Dict:
        """Get statistics about the selection process"""
        total_survivors = sum(len(r["survivors"]) for r in results.values())
        total_parents = sum(len(r["reproduction_parents"]) for r in results.values())
        total_quota = sum(r["quota"]["offspring"] for r in results.values())

        return {
            "total_survivors": total_survivors,
            "total_reproduction_parents": total_parents,
            "total_offspring_quota": total_quota,
            "species_count": len(results),
            "average_survivors_per_species": total_survivors / len(results)
            if results
            else 0,
            "average_quota_per_species": total_quota / len(results) if results else 0,
        }


if __name__ == "__main__":
    # Test selection
    print("Testing NEAT Selection...")

    # Create test networks with fitness
    from network import Network

    networks = []
    for i in range(30):
        network = Network(num_inputs=3, num_outputs=2)
        network.fitness = random.uniform(0, 100)  # Random fitness
        networks.append(network)

    # Create species
    from speciation import SpeciationManager

    speciation_manager = SpeciationManager()
    speciation_manager.assign_to_species(networks)

    print(
        f"Created {len(networks)} networks in {len(speciation_manager.species)} species"
    )

    # Test selection
    config = SelectionConfig()
    selection_manager = SelectionManager(config)

    # Perform selection
    results = selection_manager.select_next_generation(
        speciation_manager, population_size=25
    )

    # Print results
    stats = selection_manager.get_selection_statistics(results)
    print(f"Selection statistics: {stats}")

    for species_id, result in results.items():
        quota = result["quota"]
        print(
            f"Species {species_id}: {len(result['survivors'])} survivors, "
            f"{len(result['reproduction_parents'])} parents, "
            f"quota: {quota['offspring']} offspring ({quota['new_individuals']} new)"
        )

    # Test different selection methods
    print("\nTesting selection methods...")

    test_species = speciation_manager.species[0] if speciation_manager.species else None
    if test_species and len(test_species.members) >= 5:
        print(f"Testing with species of {len(test_species.members)} members")

        # Tournament selection
        config.selection_method = "tournament"
        parents = select_parents(test_species, 3, config)
        print(f"Tournament selection: {len(parents)} parents selected")

        # Roulette wheel selection
        config.selection_method = "roulette"
        parents = select_parents(test_species, 3, config)
        print(f"Roulette wheel selection: {len(parents)} parents selected")

        # Rank selection
        config.selection_method = "rank"
        parents = select_parents(test_species, 3, config)
        print(f"Rank selection: {len(parents)} parents selected")

    print("Selection testing complete!")

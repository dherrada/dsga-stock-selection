import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Set, Tuple
from src.individual import Individual

class DSGA:
    """
    - Maintains "species" via a distance threshold so seeds are diverse.
    - 'Tabu list' stores converged individuals with their selected (stock, week) sets
        to affect penalty in subsequent generations.
    """
    def __init__(self, N: int, NG: int, M: float, IS: float, SD: float, RLC: int, CL: int, W: int, stock_data: pd.DataFrame) -> None:
        self.pop_size = N
        self.num_generations = NG
        self.mutation_rate = M
        self.seed_radius = IS
        self.radius_delta = SD
        self.reeval_loop_count = RLC
        self.convergence_lim = CL
        self.weight = W
        self.stock_data = stock_data
        self.tabu_list: list[tuple[Individual, set[tuple[str, str]]]] = []
        self.population: list[Individual] = self.starting_pop()

    def starting_pop(self) -> list[Individual]:
        return [Individual() for _ in range(self.pop_size)]
    
    def dsga(self) -> list[Individual]:
        for generation in range(self.num_generations):
            for _ in range(self.reeval_loop_count):
                for ind in self.population:
                    ind.fitness_function(self.tabu_list, self.stock_data, self.weight)
                
                selected_seeds = self.seed_selection(self.population)
                children_needed = self.pop_size - len(selected_seeds)
                parents = self.parent_selection(self.population, (12 * self.pop_size) // 100)
                children = []
                
                while len(children) < children_needed:
                    child = self.crossover(parents)
                    self.mutate(child)
                    children.append(child)
                
                self.population = selected_seeds + children

            selected_seeds = self.seed_selection(self.population)

            # Convergence detection: if ≥ CL identical genomes, mark as converged and add to tabu.
            converged_inds = self.check_convergence(self.population)
            for bitstring in converged_inds:
                duplicate_inds = [ind for ind in self.population if ''.join(map(str, ind.individual)) == bitstring]
                self.tabu_list.append((duplicate_inds[0], duplicate_inds[0].get_sw_pairs(self.stock_data)))
                for ind in duplicate_inds[1:]:
                    change_idx = self.population.index(ind)
                    self.population[change_idx] = Individual()

            self.update_tabu_list(selected_seeds)
            self.seed_radius += self.radius_delta

        classification_rules = [ind for ind, _ in self.tabu_list]
        return classification_rules

    def seed_selection(self, population: list[Individual]) -> list[Individual]:
        selected_seeds = []
        sorted_pop = sorted(population, key=lambda x: x.shared_fitness, reverse=True)
        for ind in sorted_pop:
            if all(Individual.distance(ind.get_sw_pairs(self.stock_data), seed.get_sw_pairs(self.stock_data)) > self.seed_radius for seed in selected_seeds):
                selected_seeds.append(ind)
        return selected_seeds
    
    def parent_selection(self, population: list[Individual], size: int) -> list[Individual]:
        # Tournament selection
        parents = []
        for _ in range(len(population)):
            possible_parents = np.random.choice(population, size, replace=False)
            winning_parent = max(possible_parents, key=lambda x: x.shared_fitness)
            parents.append(winning_parent)
        return parents

    @staticmethod
    def crossover(parents: list[Individual]) -> Individual:
        """
        One-point style crossover on a subset of indices.
        - Randomly choose base and other parent (50/50).
        - For each index in crossover_idx where parents differ, copy other’s bit with p=0.3.
        Returns a new Individual with the mixed genome.
        """
        p1, p2 = np.random.choice(parents, 2, replace=True)

        if np.random.rand() < 0.5:
            base, other = p1, p2
        else:
            base, other = p2, p1

        crossover_idx = [1, 2, 12, 13, 23, 24, 33, 34]
        child_rule = base.individual.copy()

        for idx in crossover_idx:
            if base.individual[idx] != other.individual[idx]:
                if np.random.rand() < 0.3:
                    child_rule[idx] = other.individual[idx]
        
        child = Individual()
        child.individual = child_rule
        return child

    def mutate(self, individual: Individual) -> None:
        """
        Bit-flip mutation applied independently to each bit with probability = mutation_rate.
        """
        for idx in range(len(individual.individual)):
            if np.random.rand() < self.mutation_rate:
                individual.individual[idx] = 1 - individual.individual[idx]

    def check_convergence(self, population: list[Individual]) -> dict[str, int]:
        string_pop = [''.join(map(str, ind.individual)) for ind in population]
        ind_count = Counter(string_pop)
        converged_inds = {bitstring: count for bitstring, count in ind_count.items() if count >= self.convergence_lim}
        return converged_inds

    def update_tabu_list(self, selected_seeds: list[Individual]) -> None:
        """
        Add new seeds to tabu_list if their genome (bitstring) is not already present.
        """
        current_bitstrings = [''.join(map(str, ind.individual)) for ind, _ in self.tabu_list]
        for seed in selected_seeds:
            seed_bitstring = ''.join(map(str, seed.individual))
            if seed_bitstring not in current_bitstrings:
                self.tabu_list.append((seed, seed.get_sw_pairs(self.stock_data)))

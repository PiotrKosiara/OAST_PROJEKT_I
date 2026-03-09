from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


Chromosome = List[List[int]]


@dataclass
class EvaluationResult:
    objective: int
    loads: List[int]
    overloads: List[int] | None = None
    modules: List[int] | None = None
    capacities: List[int] | None = None


@dataclass
class Individual:
    chromosome: Chromosome
    evaluation: EvaluationResult


@dataclass
class EAConfig:
    population_size: int = 20  # N
    offspring_pairs: int = 10  # K
    mutation_probability: float = 0.1  # p
    gene_mutation_probability: float = 0.1  # q
    generations: int = 100
    seed: int | None = None


# Ogólne funkcje pomocnicze
def create_rng(seed: int | None = None) -> random.Random:
    return random.Random(seed)


def clone_chromosome(chromosome: Chromosome) -> Chromosome:
    return [gene[:] for gene in chromosome]


def chromosome_to_string(chromosome: Chromosome) -> str:
    return "[" + ", ".join(str(gene) for gene in chromosome) + "]"


# Reprezentacja rozwiązania
def random_gene(demand_volume: int, path_count: int, rng: random.Random) -> List[int]:
    """
    Losowy podział całkowitego demandu na dostępne ścieżki.
    Zwraca listę długości path_count o nieujemnych wartościach całkowitych,
    których suma daje demand_volume.
    """
    if path_count == 1:
        return [demand_volume]

    cuts = sorted(rng.randint(0, demand_volume) for _ in range(path_count - 1))
    values: List[int] = []
    prev = 0
    for cut in cuts:
        values.append(cut - prev)
        prev = cut
    values.append(demand_volume - prev)
    return values


def random_chromosome(problem_data: Dict[str, Any], rng: random.Random) -> Chromosome:
    chromosome: Chromosome = []
    for demand in problem_data["demands"]:
        chromosome.append(random_gene(demand["volume"], len(demand["paths"]), rng))
    return chromosome


# Obliczanie obciążeń i celu
def compute_link_loads(chromosome: Chromosome, problem_data: Dict[str, Any]) -> List[int]:
    loads = [0 for _ in range(problem_data["num_links"])]
    for d_idx, gene in enumerate(chromosome):
        demand = problem_data["demands"][d_idx]
        for p_idx, flow in enumerate(gene):
            if flow == 0:
                continue
            for link_id in demand["paths"][p_idx]:
                loads[link_id - 1] += flow

    return loads


def evaluate_dap(chromosome: Chromosome, problem_data: Dict[str, Any]) -> EvaluationResult:
    loads = compute_link_loads(chromosome, problem_data)
    capacities = problem_data["link_capacities"]
    overloads = [load - cap for load, cap in zip(loads, capacities)]
    objective = max(overloads)
    return EvaluationResult(
        objective=objective,
        loads=loads,
        overloads=overloads,
        capacities=capacities[:],
    )


def evaluate_ddap(chromosome: Chromosome, problem_data: Dict[str, Any]) -> EvaluationResult:
    loads = compute_link_loads(chromosome, problem_data)
    module_capacity = problem_data["module_capacity"]
    module_costs = problem_data["link_module_costs"]
    modules = [math.ceil(load / module_capacity) for load in loads]
    capacities = [m * module_capacity for m in modules]
    objective = sum(cost * module for cost, module in zip(module_costs, modules))
    return EvaluationResult(
        objective=objective,
        loads=loads,
        modules=modules,
        capacities=capacities,
    )


def evaluate_chromosome(
    chromosome: Chromosome,
    problem_data: Dict[str, Any],
    problem_type: str,
) -> EvaluationResult:
    if problem_type.upper() == "DAP":
        return evaluate_dap(chromosome, problem_data)
    if problem_type.upper() == "DDAP":
        return evaluate_ddap(chromosome, problem_data)
    raise ValueError(f"Nieznany typ problemu: {problem_type}")


# Operatory genetyczne
def crossover(parent_a: Chromosome, parent_b: Chromosome, rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    """
    Krzyżowanie: dla każdego genu losujemy, który potomek dziedziczy gen z którego rodzica.
    """
    child_a: Chromosome = []
    child_b: Chromosome = []

    for gene_a, gene_b in zip(parent_a, parent_b):
        if rng.random() < 0.5:
            child_a.append(gene_a[:])
            child_b.append(gene_b[:])
        else:
            child_a.append(gene_b[:])
            child_b.append(gene_a[:])

    return child_a, child_b


def mutate_gene(gene: List[int], rng: random.Random) -> List[int]:
    """
    Mutacja genu: przesunięcie jednej jednostki przepływu z losowo wybranej
    ścieżki o dodatnim przepływie na losowo wybraną ścieżkę.
    """
    positive_paths = [idx for idx, value in enumerate(gene) if value > 0]
    if not positive_paths:
        return gene[:]

    source = rng.choice(positive_paths)
    target = rng.randrange(len(gene))

    mutated = gene[:]
    mutated[source] -= 1
    mutated[target] += 1
    return mutated


def mutate_chromosome(
    chromosome: Chromosome,
    mutation_probability: float,
    gene_mutation_probability: float,
    rng: random.Random,
) -> Chromosome:
    mutated = clone_chromosome(chromosome)

    if rng.random() >= mutation_probability:
        return mutated

    for idx, gene in enumerate(mutated):
        if rng.random() < gene_mutation_probability:
            mutated[idx] = mutate_gene(gene, rng)

    return mutated


# Populacja i selekcja
def individual_key(individual: Individual) -> Tuple[int, str]:
    return individual.evaluation.objective, chromosome_to_string(individual.chromosome)


def sort_population(population: List[Individual]) -> List[Individual]:
    return sorted(population, key=individual_key)


def initialize_population(
    problem_data: Dict[str, Any],
    problem_type: str,
    population_size: int,
    rng: random.Random,
) -> List[Individual]:
    population: List[Individual] = []
    for _ in range(population_size):
        chromosome = random_chromosome(problem_data, rng)
        evaluation = evaluate_chromosome(chromosome, problem_data, problem_type)
        population.append(Individual(chromosome=chromosome, evaluation=evaluation))
    return sort_population(population)


def select_best_n(population: List[Individual], n: int) -> List[Individual]:
    ordered = sort_population(population)
    return ordered[:n]


def select_parents_random(population: List[Individual], rng: random.Random) -> Tuple[Individual, Individual]:
    return rng.choice(population), rng.choice(population)


# Główny algorytm EA (N+K)
def run_ea(
    problem_data: Dict[str, Any],
    problem_type: str,
    config: EAConfig,
) -> Dict[str, Any]:
    rng = create_rng(config.seed)
    population = initialize_population(
        problem_data=problem_data,
        problem_type=problem_type,
        population_size=config.population_size,
        rng=rng,
    )

    best_history = [population[0].evaluation.objective]
    avg_history = [sum(ind.evaluation.objective for ind in population) / len(population)]
    # Kryterium stopu (ustalona liczba generacji)
    # kryterium stopu = osiągnięcie config.generations iteracji pętli
    for _generation in range(config.generations):
        offspring: List[Individual] = []

        for _ in range(config.offspring_pairs):
            parent_a, parent_b = select_parents_random(population, rng)
            child_a, child_b = crossover(parent_a.chromosome, parent_b.chromosome, rng)

            child_a = mutate_chromosome(
                child_a,
                config.mutation_probability,
                config.gene_mutation_probability,
                rng,
            )
            child_b = mutate_chromosome(
                child_b,
                config.mutation_probability,
                config.gene_mutation_probability,
                rng,
            )

            offspring.append(
                Individual(
                    chromosome=child_a,
                    evaluation=evaluate_chromosome(child_a, problem_data, problem_type),
                )
            )
            offspring.append(
                Individual(
                    chromosome=child_b,
                    evaluation=evaluate_chromosome(child_b, problem_data, problem_type),
                )
            )

        population = select_best_n(population + offspring, config.population_size)
        best_history.append(population[0].evaluation.objective)
        avg_history.append(sum(ind.evaluation.objective for ind in population) / len(population))

    best = population[0]
    return {
        "best_individual": best,
        "population": population,
        "best_history": best_history,
        "avg_history": avg_history,
        "config": config,
        "problem_type": problem_type,
    }


# Formatowanie wyników
def format_solution(best: Individual, problem_type: str) -> str:
    lines: List[str] = []
    lines.append(f"Funkcja celu: {best.evaluation.objective}")
    lines.append("Przepływy ścieżkowe:")
    for idx, gene in enumerate(best.chromosome, start=1):
        lines.append(f"  d={idx}: {gene}")

    lines.append(f"Obciążenia krawędzi: {best.evaluation.loads}")

    if problem_type.upper() == "DAP":
        lines.append(f"Przeciążenia krawędzi: {best.evaluation.overloads}")
        lines.append(f"Pojemności krawędzi: {best.evaluation.capacities}")
    else:
        lines.append(f"Liczba modułów na krawędziach: {best.evaluation.modules}")
        lines.append(f"Pojemności krawędzi: {best.evaluation.capacities}")

    return "\n".join(lines)

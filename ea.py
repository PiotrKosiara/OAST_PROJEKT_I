from __future__ import annotations

"""
Algorytm ewolucyjny do rozwiązywania problemów projektowania sieci:
- DAP (Delay Allocation Problem),
- DDAP (Discrete Delay Allocation Problem).

Algorytm wykorzystuje strategię ewolucyjną typu (N + K), w której:
- N oznacza wielkość populacji,
- K oznacza liczbę par rodziców generujących potomstwo.

Każde rozwiązanie jest reprezentowane jako "chromosom" opisujący podział
natężenia ruchu pomiędzy dostępne ścieżki dla każdego zapotrzebowania.

Poszczególne pliki zawierają:
- reprezentację rozwiązania (geny i chromosomy),
- funkcje oceny dla problemów DAP i DDAP,
- operatory genetyczne (krzyżowanie i mutację),
- operacje na populacji,
- implementację głównej pętli algorytmu ewolucyjnego.
"""

import random
from typing import Any, Dict, List

from evaluation import evaluate_chromosome
from models import Chromosome, EAConfig, Individual
from operators import crossover, mutate_chromosome
from population import (
    insert_sorted,
    merge_best_n,
    population_objective_sum,
    population_size,
    select_parent_pair,
)

def create_rng(seed: int | None = None) -> random.Random:
    """Generator liczb losowych."""
    return random.Random(seed)

def random_gene(demand_volume: int, path_count: int, rng: random.Random) -> List[int]:
    """
    Losowy podział całkowitego jednego zapotrzebowania (demand) na dostępne ścieżki.

    Suma wartości w genie jest równa całkowitemu jednemu zapotrzebowaniu,
    a każda wartość odpowiada przepływowi przepisanemu do danej ścieżki.

    Arguments:
        demand_volume: całkowite zapotrzebowanie ruchu
        path_count: liczba dostępnych ścieżek
        rng: generator liczb losowych

    Returns:
        Lista długości path_count o nieujemnych wartościach całkowitych, których 
        suma daje demand_volume.
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
    """
    Generowanie losowego chromosomu będącego rozwiązaniem

    Każdy gen to jedno zapotrzebowanie (demand) i opisuje podział ruchu 
    pomiędzy dostępne ścieżki

    Arguments:
        problem_data: dane danego zadania - problemu "ścieżek"
        rng: generator liczb losowych

    Returns:
        Losowy wygenerowany chromosom
    """
    chromosome: Chromosome = []
    for demand in problem_data["demands"]:
        chromosome.append(random_gene(demand["volume"], len(demand["paths"]), rng))
    return chromosome


def initialize_population(
    problem_data: Dict[str, Any],
    problem_type: str,
    population_size: int,
    rng: random.Random,
):
    head = None

    for _ in range(population_size):
        chromosome = random_chromosome(problem_data, rng)
        evaluation = evaluate_chromosome(
            chromosome,
            problem_data,
            problem_type,
        )
        individual = Individual(
            chromosome=chromosome,
            evaluation=evaluation
        )

        head = insert_sorted(head, individual)

    return head

def run_ea(
    problem_data: Dict[str, Any],
    problem_type: str,
    config: EAConfig,
) -> Dict[str, Any]:
    """
    Algorytm ewolucyjny.

    Algorytm działa według strategii (N + K): 
    populacja o wielkości N generuje potomstwo,
    a następnie wybierane jest najlepsze N osobników
    z populacji rodziców i potomstwa.

    Arguments:
        problem_data: dane danego zadania - problemu "ścieżek"
        problem_type: typ problemu (DAP v DDAP)
        config: konfiguracja algorytmu

    Returns:
        Słownik zawierający najlepsze znalezione rozwiązanie,
        końcową populację oraz historię wartości funkcji celu

    W każdej generacji:
    1. losujemy pary rodziców z populacji P(n),
    2. tworzymy potomstwo O,
    3. łączymy P(n) i O,
    4. zostawiamy N najlepszych osobników.
    """
    rng = create_rng(config.seed)
    population = initialize_population(
        problem_data=problem_data,
        problem_type=problem_type,
        population_size=config.population_size,
        rng=rng,
    )

    best_history = [population.individual.evaluation.objective]
    avg_history = [
        population_objective_sum(population) /
        population_size(population)
    ]
   
    for _generation in range(config.generations):
        offspring = None

        for _ in range(config.offspring_pairs):
            parent_a, parent_b = select_parent_pair(
                population,
                config.parent_selection_method,
                rng,
            )
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

            offspring = insert_sorted(
                offspring,
                Individual(
                    chromosome=child_a,
                    evaluation=evaluate_chromosome(child_a, problem_data, problem_type)
                )
            )

            offspring = insert_sorted(
                offspring,
                Individual(
                    chromosome=child_b,
                    evaluation=evaluate_chromosome(child_b, problem_data, problem_type)
                )
            )
            
        population = merge_best_n(
            population,
            offspring,
            config.population_size
        )
        best_history.append(population.individual.evaluation.objective)
        avg_history.append(
            population_objective_sum(population) / population_size(population)
        )


    best = population.individual
    return {
        "best_individual": best,
        "population": population,
        "best_history": best_history,
        "avg_history": avg_history,
        "config": config,
        "problem_type": problem_type,
    }

# ====================
# Formatowanie wyników
# ====================

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
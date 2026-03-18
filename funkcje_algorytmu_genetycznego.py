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

Moduł zawiera:
- reprezentację rozwiązania (geny i chromosomy),
- funkcje oceny dla problemów DAP i DDAP,
- operatory genetyczne (krzyżowanie i mutację),
- operacje na populacji,
- implementację głównej pętli algorytmu ewolucyjnego.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Optional


Chromosome = List[List[int]]


@dataclass
class EvaluationResult:
    """
    Wynik oceny chromosomu

    Atrybuty:
        objective: wartość funkcji celu
        loads: obciążenia poszczególnych krawędzi
        overloads: wartości przeciążeń krawędzi (DAP)
        modules: liczba modułów na krawędziach (DDAP)
        capacities: pojemności krawędzi
    """

    objective: int
    loads: List[int]
    overloads: List[int] | None = None
    modules: List[int] | None = None
    capacities: List[int] | None = None


@dataclass
class Individual:
    """
    Pojedynczy osobnik w populacji

    Atrybuty:
        chromosome: chromosom 
        evaluation: wynik oceny chromosomu???
    """
    chromosome: Chromosome
    evaluation: EvaluationResult


@dataclass
class EAConfig:
    """
    Parametry algorytmu ewolucyjnego

    Atrybuty:
        population_size: liczba osobników w populacji (N)
        offspring_pairs: liczba par rodziców (K)
        mutation_probability: prawdopodobieństwo mutacji chromosomu (p)
        gene_mutation_probability: prawdopodobieństwo mutacji genu (q)
        generations: liczba generacji
        seed:
    """
    population_size: int = 20  # N
    offspring_pairs: int = 10  # K
    mutation_probability: float = 0.1  # p
    gene_mutation_probability: float = 0.1  # q
    generations: int = 100
    seed: int | None = None

@dataclass
class Node:
    """Element jednokierunkowej, uporządkowanej listy osobników."""
    def __init__(self, individual):
        self.individual = individual
        self.next = None


# Ogólne funkcje pomocnicze

def create_rng(seed: int | None = None) -> random.Random:
    """Generator liczb losowych."""
    return random.Random(seed)


def clone_chromosome(chromosome: Chromosome) -> Chromosome:
    """Kopia chromosomu."""
    return [gene[:] for gene in chromosome]


def chromosome_to_string(chromosome: Chromosome) -> str:
    """Konwersja chromosomu do reprezentacji tekstowej."""
    return "[" + ", ".join(str(gene) for gene in chromosome) + "]"

# =========================
# Reprezentacja rozwiązania
# =========================

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

# ==========================
# Obliczanie obciążeń i celu
# ==========================

def compute_link_loads(chromosome: Chromosome, problem_data: Dict[str, Any]) -> List[int]:
    """
    Obliczanie obciążeń krawędzi

    Arguments:
        chromosome: chromosom - rozwiązanie
        problem_data: dane danego zadania - problemu "ścieżek"

    Returns:
        Lista obciążeń dla każdej krawędzi
    """
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
    """
    Obliczanie wartości funkcji celu dla problemu DAP

    Arguments:
        chromosome: -_-
        problem_data: dane danego zadania - problemu "ścieżek"

    Returns:
        Wynik rozwiązania
    """
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
    """
    Obliczanie wartości funkcji celu dla problemu DDAP

    Arguments:
        chromosome: -_-
        problem data: dane danego zadania - problemu "ścieżek"

    Returns:
        Wynik rozwiązania
    """
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
    """
    Ocena chromosomu w zależności od typu problemu

    Arguments:
        chromosome: -_-
        problem_data: dane danego zadania - problemu "ścieżek"
        problem_type: typ problemu (DAP v DDAP)

    Returns:
        Wynik oceny chromosomu
    """
    if problem_type.upper() == "DAP":
        return evaluate_dap(chromosome, problem_data)
    if problem_type.upper() == "DDAP":
        return evaluate_ddap(chromosome, problem_data)
    raise ValueError(f"Nieznany typ problemu: {problem_type}")

# ====================
# Operatory genetyczne
# ====================

def crossover(parent_a: Chromosome, parent_b: Chromosome, rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    """
    Krzyżowanie dwóch chromosomów. Dla każdego genu losujemy, który potomek dziedziczy 
    gen z którego rodzica

    Arguments:
        parent_a: chromosom pierwszego rodzica
        parent_b: chormosom drugiego rodzica
        rng: generator liczb losowych

    Returns:
        Dwa nowe chromosowy potomne
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
    Mutacja pojedynczego genu

    Przesunięcie jednej jednostki przepływu z losowo wybranej ścieżki 
    o dodatnim przepływie na losowo wybraną ścieżkę

    Arguments:
        gene:
        rng: generator liczb losowych

    Returns: 
        Zmutowany gen
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
    """
    Mutacja chromosomu

    Mutacja jest wykonywana z określonym prawdopodobieństwem,
    a następnie każdy gen może zostać zmutowany niezależnie

    Arguments:
        chromosome: chromosom do mutacji
        mutation_probability: prawdopodobieństwo mutacji chromosomu
        gene_mutation_probability: prawdopodobieństwo mutacji genu
        rng: generator liczb losowych    

    Returns:
        Zmutowany chromosom   
    """
    mutated = clone_chromosome(chromosome)

    if rng.random() >= mutation_probability:
        return mutated

    for idx, gene in enumerate(mutated):
        if rng.random() < gene_mutation_probability:
            mutated[idx] = mutate_gene(gene, rng)

    return mutated

# ====================
# Populacja i selekcja
# ====================

def individual_key(individual: Individual) -> Tuple[int, str]:
    """
    Klucz porządkujący osobników.

    Najpierw porównywane są wartość funkcji celu, a potem tekstową postać
    chromosomu.
    """
    return individual.evaluation.objective, chromosome_to_string(individual.chromosome)

def iterate_population(head: Optional[Node]) -> Iterable[Individual]:
    """Iteacja po osobnikach zapisanych w linked list."""
    current = head
    while current is not None:
        yield current.individual
        current = current.next

def population_size(head: Optional[Node]) -> int:
    """Zwraca liczbę osobników w populacji."""
    return sum(1 for _ in iterate_population(head))


def population_objective_sum(head: Optional[Node]) -> int:
    """Zwraca sumę wartości funkcji celu w populacji."""
    return sum(ind.evaluation.objective for ind in iterate_population(head))


def get_random_individual(head: Optional[Node], rng: random.Random) -> Individual:
    """Losuje jednego osobnika z linked list."""
    size = population_size(head)

    chosen_index = rng.randrange(size)
    for index, individual in enumerate(iterate_population(head)):
        if index == chosen_index:
            return individual


def insert_sorted(head: Optional[Node], individual: Individual) -> Node:
    """
    Wstawia osobnika do uporządkowanej linked list.

    Lista jest uporządkowana rosnąco względem wartości funkcji celu.
    """
    new_node = Node(individual=individual)

    if head is None or individual_key(individual) < individual_key(head.individual):
        new_node.next = head
        return new_node

    current = head
    while current.next is not None and individual_key(current.next.individual) < individual_key(individual):
        current = current.next

    new_node.next = current.next
    current.next = new_node
    return head


def merge_best_n(head_a: Optional[Node], head_b: Optional[Node], n: int) -> Optional[Node]:
    """Scala dwie już uporządkowane populacje i zwraca pierwsze n najlepszych osobników."""
    if n <= 0:
        return None

    result_head: Optional[Node] = None
    result_tail: Optional[Node] = None
    added = 0

    a = head_a
    b = head_b

    while added < n and (a is not None or b is not None):
        if b is None or (a is not None and individual_key(a.individual) <= individual_key(b.individual)):
            chosen = a.individual
            a = a.next if a is not None else None
        else:
            chosen = b.individual
            b = b.next if b is not None else None

        new_node = Node(individual=chosen)
        if result_head is None:
            result_head = new_node
            result_tail = new_node
        else:
            result_tail.next = new_node
            result_tail = new_node

        added += 1

    return result_head


# ======================================
# Inicjalizacja Główny algorytm EA (N+K)
# ======================================

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
            parent_a = get_random_individual(population, rng)
            parent_b = get_random_individual(population, rng)
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
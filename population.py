import random
from typing import Iterable, Optional, Tuple, List

from models import Individual, Node, Chromosome

def chromosome_to_string(chromosome: Chromosome) -> str:
    """Konwersja chromosomu do reprezentacji tekstowej."""
    return "[" + ", ".join(str(gene) for gene in chromosome) + "]"


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


def population_to_list(head: Optional[Node]) -> List[Individual]:
    """Zamienia uporządkowaną linked listę populacji na zwykłą listę."""
    return list(iterate_population(head))


def build_pd13_probabilities(population_list: List[Individual]) -> List[float]:
    """
    Buduwanie rozkładu p(n).

    Dla populacji y(1),...,y(N) uporządkowanej niemalejąco względem F(y):
        F* = max F(y(n))
        S = sum(F* + 1 - F(y(n)))
        p(n) = (F* + 1 - F(y(n))) / S
    """
    if not population_list:
        raise ValueError("Populacja nie może być pusta.")

    objectives = [ind.evaluation.objective for ind in population_list]
    f_star = max(objectives)
    weights = [f_star + 1 - value for value in objectives]
    s = sum(weights)

    return [weight / s for weight in weights]


def choose_individual_by_probability(
    population_list: List[Individual],
    probabilities: List[float],
    rng: random.Random,
) -> Individual:
    """Losuje osobnika zgodnie z zadanym rozkładem prawdopodobieństwa."""
    r = rng.random()
    cumulative = 0.0

    for individual, probability in zip(population_list, probabilities):
        cumulative += probability
        if r <= cumulative:
            return individual

    return population_list[-1]

def select_parent_pair(
    population: Optional[Node],
    method: str,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """
    Wybór pary rodziców do krzyżowania.

    Metody:
    - random_random : obaj rodzice losowo z populacji
    - best_pn       : pierwszy rodzic = najlepszy, drugi wg p(n)
    - pn_pn         : obaj rodzice wg p(n)
    """
    population_list = population_to_list(population)

    if not population_list:
        raise ValueError("Nie można wybrać rodziców z pustej populacji.")

    if method == "random_random":
        return rng.choice(population_list), rng.choice(population_list)

    probabilities = build_pd13_probabilities(population_list)

    if method == "best_pn":
        return population_list[0], choose_individual_by_probability(
            population_list,
            probabilities,
            rng,
        )

    if method == "pn_pn":
        return (
            choose_individual_by_probability(population_list, probabilities, rng),
            choose_individual_by_probability(population_list, probabilities, rng),
        )

    raise ValueError(f"Nieznana metoda doboru rodziców: {method}")

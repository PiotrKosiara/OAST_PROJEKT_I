import random
from typing import Iterable, Optional, Tuple

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



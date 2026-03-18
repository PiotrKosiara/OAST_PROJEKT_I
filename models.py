from __future__ import annotations

from dataclasses import dataclass
from typing import List


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
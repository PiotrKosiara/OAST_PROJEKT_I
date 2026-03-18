import random
from typing import List, Tuple

from models import Chromosome


def clone_chromosome(chromosome: Chromosome) -> Chromosome:
    """Kopia chromosomu."""
    return [gene[:] for gene in chromosome]

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
    o dodatnim (bo odejmując z source, nie może wyjść ujemnie) przepływie na losowo wybraną ścieżkę

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
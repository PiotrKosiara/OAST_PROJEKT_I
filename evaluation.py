import math
from typing import Any, Dict, List

from models import Chromosome, EvaluationResult

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
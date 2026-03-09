from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt


# Wczytywanie danych z plików
def _extract_int(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"Nie udało się odczytać pola: {label}")
    return int(match.group(1))


def _extract_block(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Nie udało się odczytać bloku: {label}")
    return match.group(1)


def parse_dap_file(file_path: str | Path) -> Dict[str, Any]:
    text = Path(file_path).read_text(encoding="utf-8")

    max_node = _extract_int(r"param\s+maxNode\s*:=\s*(\d+)\s*;", text, "maxNode")
    module_capacity = _extract_int(
        r"param\s+moduleCapacity\s*:=\s*(\d+)\s*;", text, "moduleCapacity"
    )

    links_block = _extract_block(
        r"param:\s*Links:.*?:=\s*(.*?)\s*;", text, "Links"
    )
    demands_block = _extract_block(
        r"param:\s*Demands:.*?:=\s*(.*?)\s*;", text, "Demands"
    )

    link_capacities: List[int] = []
    for line in links_block.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        link_capacities.append(int(parts[3]))

    demands: List[Dict[str, Any]] = []
    for line in demands_block.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        demands.append(
            {
                "id": int(parts[0]),
                "source": int(parts[1]),
                "target": int(parts[2]),
                "volume": int(parts[3]),
                "max_paths": int(parts[4]),
                "paths": [],
            }
        )

    path_matches = re.findall(
        r"set\s+DemandPath_links\[(\d+),(\d+)\]\s*:=\s*(.*?)\s*;",
        text,
        re.MULTILINE | re.DOTALL,
    )

    demand_path_map: Dict[int, Dict[int, List[int]]] = {}
    for d_str, p_str, path_text in path_matches:
        d = int(d_str)
        p = int(p_str)
        links = [int(x) for x in path_text.split()]
        demand_path_map.setdefault(d, {})[p] = links

    for demand in demands:
        d_id = demand["id"]
        demand["paths"] = [
            demand_path_map[d_id][path_idx]
            for path_idx in sorted(demand_path_map[d_id].keys())
        ]

    return {
        "problem_name": "DAP",
        "num_nodes": max_node,
        "num_links": len(link_capacities),
        "module_capacity": module_capacity,
        "link_capacities": link_capacities,
        "demands": demands,
    }


def parse_ddap_file(file_path: str | Path) -> Dict[str, Any]:
    text = Path(file_path).read_text(encoding="utf-8")

    max_node = _extract_int(r"param\s+maxNode\s*:=\s*(\d+)\s*;", text, "maxNode")
    module_capacity = _extract_int(
        r"param\s+moduleCapacity\s*:=\s*(\d+)\s*;", text, "moduleCapacity"
    )

    links_block = _extract_block(
        r"param:\s*Links:.*?:=\s*(.*?)\s*;", text, "Links"
    )
    demands_block = _extract_block(
        r"param:\s*Demands:.*?:=\s*(.*?)\s*;", text, "Demands"
    )

    link_module_costs: List[int] = []
    for line in links_block.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        link_module_costs.append(int(parts[3]))

    demands: List[Dict[str, Any]] = []
    for line in demands_block.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        demands.append(
            {
                "id": int(parts[0]),
                "source": int(parts[1]),
                "target": int(parts[2]),
                "volume": int(parts[3]),
                "max_paths": int(parts[4]),
                "paths": [],
            }
        )

    path_matches = re.findall(
        r"set\s+Demand_pathLinks\[(\d+),(\d+)\]\s*:=\s*(.*?)\s*;",
        text,
        re.MULTILINE | re.DOTALL,
    )

    demand_path_map: Dict[int, Dict[int, List[int]]] = {}
    for d_str, p_str, path_text in path_matches:
        d = int(d_str)
        p = int(p_str)
        links = [int(x) for x in path_text.split()]
        demand_path_map.setdefault(d, {})[p] = links

    for demand in demands:
        d_id = demand["id"]
        demand["paths"] = [
            demand_path_map[d_id][path_idx]
            for path_idx in sorted(demand_path_map[d_id].keys())
        ]

    return {
        "problem_name": "DDAP",
        "num_nodes": max_node,
        "num_links": len(link_module_costs),
        "module_capacity": module_capacity,
        "link_module_costs": link_module_costs,
        "demands": demands,
    }


# Wykresy do sprawozdania
def plot_best_history(best_history: List[float], title: str, save_path: str | Path | None = None) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(best_history)), best_history, marker="o", linewidth=1)
    plt.xlabel("Generacja")
    plt.ylabel("Najlepsza wartość funkcji celu")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def summarize_instance(problem_data: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Liczba węzłów: {problem_data['num_nodes']}")
    lines.append(f"Liczba łączy: {problem_data['num_links']}")
    lines.append(f"Liczba żądań: {len(problem_data['demands'])}")
    lines.append(f"Pojemność modułu: {problem_data['module_capacity']}")
    return "\n".join(lines)


# Prosta weryfikacja brute force dla sieci 4-węzłowej
def _compositions(n: int, k: int):
    if k == 1:
        yield [n]
        return
    for first in range(n + 1):
        for rest in _compositions(n - first, k - 1):
            yield [first] + rest


def brute_force_verify(problem_data: Dict[str, Any], evaluator) -> Dict[str, Any]:
    from itertools import product

    all_genes = [
        list(_compositions(demand["volume"], len(demand["paths"])))
        for demand in problem_data["demands"]
    ]

    total_patterns = 1
    for genes in all_genes:
        total_patterns *= len(genes)

    best_chromosome = None
    best_evaluation = None

    for chromosome_tuple in product(*all_genes):
        chromosome = [list(gene) for gene in chromosome_tuple]
        evaluation = evaluator(chromosome, problem_data)
        if best_evaluation is None or evaluation.objective < best_evaluation.objective:
            best_chromosome = chromosome
            best_evaluation = evaluation

    return {
        "flow_patterns_count": total_patterns,
        "best_chromosome": best_chromosome,
        "best_evaluation": best_evaluation,
    }

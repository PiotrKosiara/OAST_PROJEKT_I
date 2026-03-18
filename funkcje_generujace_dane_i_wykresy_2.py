from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from ea import run_ea
from evaluation import evaluate_dap, evaluate_ddap
from models import EAConfig
from funkcje_generujące_dane_i_wykresy_do_sprawozdania import (
    parse_dap_file,
    parse_ddap_file,
    summarize_instance,
    brute_force_verify,
)

METHOD_LABELS = {
    "random_random": "losowo-losowo",
    "best_pn": "najlepszy-p(n)",
    "pn_pn": "p(n)-p(n)",
}


def load_problem_data(problem_type: str, data_path: str | Path) -> Dict[str, Any]:
    if problem_type.upper() == "DAP":
        return parse_dap_file(data_path)
    if problem_type.upper() == "DDAP":
        return parse_ddap_file(data_path)
    raise ValueError(f"Nieobsługiwany typ problemu: {problem_type}")


def get_exact_result(problem_type: str, problem_data: Dict[str, Any]) -> Dict[str, Any]:
    evaluator = evaluate_dap if problem_type.upper() == "DAP" else evaluate_ddap
    return brute_force_verify(problem_data, evaluator)


def run_batch_for_method(
    problem_data: Dict[str, Any],
    problem_type: str,
    method: str,
    runs_count: int,
    population_size: int,
    offspring_pairs: int,
    mutation_probability: float,
    gene_mutation_probability: float,
    generations: int,
    base_seed: int | None = None,
) -> Dict[str, Any]:
    best_histories: List[List[float]] = []
    avg_histories: List[List[float]] = []
    final_objectives: List[float] = []

    for run_idx in range(runs_count):
        seed = None if base_seed is None else base_seed + run_idx

        config = EAConfig(
            population_size=population_size,
            offspring_pairs=offspring_pairs,
            mutation_probability=mutation_probability,
            gene_mutation_probability=gene_mutation_probability,
            generations=generations,
            parent_selection_method=method,
            seed=seed,
        )

        result = run_ea(
            problem_data=problem_data,
            problem_type=problem_type,
            config=config,
        )

        best_histories.append(result["best_history"])
        avg_histories.append(result["avg_history"])
        final_objectives.append(result["best_individual"].evaluation.objective)

    return {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "best_histories": best_histories,
        "avg_histories": avg_histories,
        "final_objectives": final_objectives,
    }


def compute_fan_stats(histories: List[List[float]]) -> Dict[str, np.ndarray]:
    data = np.asarray(histories, dtype=float)

    return {
        "mean": np.mean(data, axis=0),
        "min": np.min(data, axis=0),
        "p10": np.percentile(data, 10, axis=0),
        "p25": np.percentile(data, 25, axis=0),
        "median": np.percentile(data, 50, axis=0),
        "p75": np.percentile(data, 75, axis=0),
        "p90": np.percentile(data, 90, axis=0),
        "max": np.max(data, axis=0),
    }

def adjust_color(color: str, factor: float) -> tuple:
    """
    Zmienia odcień koloru:
    - factor < 1.0 -> kolor ciemniejszy
    - factor > 1.0 -> kolor jaśniejszy
    """
    r, g, b = mcolors.to_rgb(color)

    if factor >= 1.0:
        r = r + (1.0 - r) * (factor - 1.0)
        g = g + (1.0 - g) * (factor - 1.0)
        b = b + (1.0 - b) * (factor - 1.0)
    else:
        r *= factor
        g *= factor
        b *= factor

    return (min(max(r, 0.0), 1.0),
            min(max(g, 0.0), 1.0),
            min(max(b, 0.0), 1.0))

def plot_single_method_fan_chart(
    histories: List[List[float]],
    title: str,
    ylabel: str,
    save_path: str | Path,
    base_color: str = "#1f77b4",
) -> None:
    stats = compute_fan_stats(histories)
    generations = range(len(stats["median"]))

    outer_color = adjust_color(base_color, 1.35)   # najjaśniejszy pas
    inner_color = adjust_color(base_color, 1.15)   # ciemniejszy pas
    median_color = adjust_color(base_color, 0.75)  # ciemna linia
    mean_color = adjust_color(base_color, 0.95)    # trochę jaśniejsza linia

    plt.figure(figsize=(9, 5))

    plt.fill_between(
        generations,
        stats["p10"],
        stats["p90"],
        color=outer_color,
        alpha=0.35,
        label="10-90 percentyl",
    )

    plt.fill_between(
        generations,
        stats["p25"],
        stats["p75"],
        color=inner_color,
        alpha=0.55,
        label="25-75 percentyl",
    )

    plt.plot(
        generations,
        stats["median"],
        color=median_color,
        linewidth=2.4,
        label="Mediana",
    )

    plt.plot(
        generations,
        stats["mean"],
        color=mean_color,
        linewidth=2.0,
        linestyle="--",
        label="Średnia",
    )

    plt.xlabel("Generacja")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_fan_plots_for_all_methods(
    method_results: List[Dict[str, Any]],
    problem_type: str,
    output_dir: str | Path,
) -> Dict[str, List[Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_paths: List[Path] = []
    avg_paths: List[Path] = []

    for result in method_results:
        method = result["method"]
        label = result["label"]

        best_path = output_dir / f"wachlarz_best_{problem_type.lower()}_{method}.png"
        avg_path = output_dir / f"wachlarz_avg_{problem_type.lower()}_{method}.png"

        plot_single_method_fan_chart(
            histories=result["best_histories"],
            title=f"{problem_type} - {label} - najlepsza wartość funkcji celu",
            ylabel="Najlepsza wartość funkcji celu",
            save_path=best_path,
        )

        plot_single_method_fan_chart(
            histories=result["avg_histories"],
            title=f"{problem_type} - {label} - średnia wartość funkcji celu",
            ylabel="Średnia wartość funkcji celu w populacji",
            save_path=avg_path,
        )

        best_paths.append(best_path)
        avg_paths.append(avg_path)

    return {
        "best_plot_paths": best_paths,
        "avg_plot_paths": avg_paths,
    }


def format_comparison_summary(
    problem_type: str,
    data_path: str | Path,
    problem_data: Dict[str, Any],
    method_results: List[Dict[str, Any]],
    exact_result: Dict[str, Any] | None = None,
) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"Porównanie metod doboru rodziców - {problem_type}")
    lines.append(f"Plik danych: {data_path}")
    lines.append(summarize_instance(problem_data))
    lines.append("")

    exact_objective = None
    if exact_result is not None and exact_result.get("best_evaluation") is not None:
        exact_objective = exact_result["best_evaluation"].objective
        lines.append(f"Dokładne optimum brute force: {exact_objective}")
        lines.append("")

    for method_result in method_results:
        values = np.asarray(method_result["final_objectives"], dtype=float)

        lines.append(f"Metoda: {method_result['label']} ({method_result['method']})")
        lines.append(f"  liczba uruchomień: {len(values)}")
        lines.append(f"  średni wynik końcowy: {values.mean():.4f}")
        lines.append(f"  mediana wyniku końcowego: {np.median(values):.4f}")
        lines.append(f"  odchylenie standardowe: {values.std():.4f}")
        lines.append(f"  min wynik końcowy: {values.min():.4f}")
        lines.append(f"  max wynik końcowy: {values.max():.4f}")

        if exact_objective is not None:
            hit_count = int(np.sum(values == exact_objective))
            lines.append(f"  liczba trafień optimum: {hit_count}/{len(values)}")

        lines.append("")

    return "\n".join(lines)


def save_text_report(text: str, save_path: str | Path) -> None:
    Path(save_path).write_text(text, encoding="utf-8")


def run_comparison_and_save(
    problem_type: str,
    data_path: str | Path,
    output_dir: str | Path,
    runs_count: int = 100,
    generations: int = 100,
    population_size: int = 20,
    offspring_pairs: int = 10,
    mutation_probability: float = 0.1,
    gene_mutation_probability: float = 0.1,
    methods: List[str] | None = None,
    base_seed: int | None = 1000,
) -> Dict[str, Any]:
    if methods is None:
        methods = ["random_random", "best_pn", "pn_pn"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_data = load_problem_data(problem_type, data_path)
    exact_result = get_exact_result(problem_type, problem_data)

    method_results: List[Dict[str, Any]] = []
    for method in methods:
        result = run_batch_for_method(
            problem_data=problem_data,
            problem_type=problem_type,
            method=method,
            runs_count=runs_count,
            population_size=population_size,
            offspring_pairs=offspring_pairs,
            mutation_probability=mutation_probability,
            gene_mutation_probability=gene_mutation_probability,
            generations=generations,
            base_seed=base_seed,
        )
        method_results.append(result)

    plot_paths = save_fan_plots_for_all_methods(
        method_results=method_results,
        problem_type=problem_type,
        output_dir=output_dir,
    )

    report_path = output_dir / f"podsumowanie_{problem_type.lower()}.txt"
    report_text = format_comparison_summary(
        problem_type=problem_type,
        data_path=data_path,
        problem_data=problem_data,
        method_results=method_results,
        exact_result=exact_result,
    )
    save_text_report(report_text, report_path)

    return {
        "problem_type": problem_type,
        "problem_data": problem_data,
        "exact_result": exact_result,
        "method_results": method_results,
        "best_plot_paths": plot_paths["best_plot_paths"],
        "avg_plot_paths": plot_paths["avg_plot_paths"],
        "report_path": report_path,
    }
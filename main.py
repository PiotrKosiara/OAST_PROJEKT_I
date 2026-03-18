from __future__ import annotations
from pathlib import Path
from ea import run_ea, format_solution
from models import EAConfig
from evaluation import evaluate_dap, evaluate_ddap
# from funkcje_algorytmu_genetycznego import EAConfig, format_solution, run_ea, evaluate_dap, evaluate_ddap
from funkcje_generujące_dane_i_wykresy_do_sprawozdania import (
    parse_dap_file,
    parse_ddap_file,
    plot_best_history,
    summarize_instance,
    brute_force_verify,
)

# Ścieżka
DIR = Path(__file__).resolve().parent


def run_single_experiment(problem_type: str, data_path: Path, output_plot: Path):
    if problem_type == "DAP":
        problem_data = parse_dap_file(data_path)
    elif problem_type == "DDAP":
        problem_data = parse_ddap_file(data_path)
    else:
        raise ValueError(f"Nieobsługiwany typ problemu: {problem_type}")

    print("=" * 80)
    print(f"Uruchomienie dla problemu: {problem_type}")
    print(f"Plik danych: {data_path}")
    print(summarize_instance(problem_data))

    # Parametry zgodne z wymaganiami etapu 1 projektu
    config = EAConfig(
        population_size=20,
        offspring_pairs=10,
        mutation_probability=0.1,
        gene_mutation_probability=0.1,
        generations=100,
        parent_selection_method="pn_pn",
        seed=404,
        # "random_random"  -> obaj rodzice losowo
        # "best_pn"        -> pierwszy najlepszy, drugi wg p(n)
        # "pn_pn"          -> obaj wg p(n)
    )

    result = run_ea(problem_data=problem_data, problem_type=problem_type, config=config)

    evaluator = evaluate_dap if problem_type == "DAP" else evaluate_ddap
    exact = brute_force_verify(problem_data, evaluator)

    print("\nNajlepsze znalezione rozwiązanie:")
    print(format_solution(result["best_individual"], problem_type))

    plot_best_history(
        result["best_history"],
        title=f"EA ({problem_type}) - najlepsza wartość funkcji celu",
        save_path=output_plot,
    )
    print(f"\nZapisano wykres trajektorii do: {output_plot}")

    print("\nWeryfikacja brute force dla sieci 4-węzłowej:")
    print(f"Liczba wszystkich wzorców przepływu: {exact['flow_patterns_count']}")
    print(f"Optymalna wartość funkcji celu: {exact['best_evaluation'].objective}")
    print(f"Czy EA znalazł optimum?: {result['best_individual'].evaluation.objective == exact['best_evaluation'].objective}")


def main():
    run_single_experiment(
        problem_type="DAP",
        data_path=DIR / "dap-net4.txt",
        output_plot=DIR / "trajektoria_dap_pn.png",
    )

    run_single_experiment(
        problem_type="DDAP",
        data_path=DIR / "ddap-net4.txt",
        output_plot=DIR / "trajektoria_ddap_pn.png",
    )


if __name__ == "__main__":
    main()

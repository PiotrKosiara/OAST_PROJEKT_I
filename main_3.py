from __future__ import annotations

from pathlib import Path

from funkcje_generujace_dane_i_wykresy_2 import run_comparison_and_save

DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DIR / "wyniki_porownawcze_net12"


def run_problem_comparison(problem_type: str, data_file_name: str) -> None:
    print("=" * 80)
    print(f"Porównanie metod doboru rodziców dla problemu: {problem_type}")

    result = run_comparison_and_save(
        problem_type=problem_type,
        data_path=DIR / data_file_name,
        output_dir=OUTPUT_DIR,
        runs_count=100,
        generations=100,
        population_size=50,
        offspring_pairs=25,
        mutation_probability=0.1,
        gene_mutation_probability=0.1,
        methods=["random_random", "best_pn", "pn_pn"],
        base_seed=1000,
    )

    print("\nZapisane wykresy best:")
    for path in result["best_plot_paths"]:
        print(f"  {path}")

    print("\nZapisane wykresy avg:")
    for path in result["avg_plot_paths"]:
        print(f"  {path}")

    print(f"\nZapisano raport: {result['report_path']}\n")
    print(Path(result["report_path"]).read_text(encoding="utf-8"))


def main():
    run_problem_comparison(
        problem_type="DAP",
        data_file_name="dap-net12-26L.txt",
    )

    run_problem_comparison(
        problem_type="DDAP",
        data_file_name="ddap-net12-26L.txt",
    )


if __name__ == "__main__":
    main()
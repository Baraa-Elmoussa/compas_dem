"""Run every benchmark case and print a combined timing/stability summary.

Usage
-----
    conda activate masonry
    python scripts/benchmarks/run_all.py

Each case is also runnable standalone, e.g. ``python case_arch_fails.py``.
"""

import case_arch_fails
import case_arch_stands
import case_armadillo
import case_barrel_vault
import case_cross_vault
import case_stack

CASES = [
    ("Stack", case_stack),
    ("Arch (stands)", case_arch_stands),
    ("Arch (fails)", case_arch_fails),
    ("Barrel vault", case_barrel_vault),
    ("Cross vault", case_cross_vault),
    ("Armadillo", case_armadillo),
]


def main() -> None:
    all_rows = {label: module.main() for label, module in CASES}

    print("\n\n" + "=" * 72)
    print("SUMMARY -- wall-clock solve time and stability verdict per case")
    print("=" * 72)
    header = f"{'Case':<16}{'Solver':<12}{'Status':<10}{'Time [s]':>10}{'Verdict':>12}"
    print(header)
    print("-" * len(header))
    for label, rows in all_rows.items():
        for row in rows:
            time_str = f"{row.elapsed:.3f}" if row.elapsed is not None else "-"
            print(f"{label:<16}{row.solver:<12}{row.status:<10}{time_str:>10}{(row.verdict or '-'):>12}")


if __name__ == "__main__":
    main()

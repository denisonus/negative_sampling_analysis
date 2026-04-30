"""Entry point for experiment analysis.

Usage:
    python analysis.py results.json                          # single-run report
    python analysis.py a.json b.json --sweep_strategy dns    # parameter sweep
"""

from _analysis.report import generate_full_report

__all__ = ["generate_full_report"]


if __name__ == "__main__":
    from _analysis.cli import main

    main()

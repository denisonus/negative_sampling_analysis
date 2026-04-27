"""Compatibility entry point for automatic experiment analysis."""

from _analysis.report import generate_full_report
from _analysis.tables import (
    build_relative_improvement_rows,
    save_relative_improvement_table,
)

__all__ = [
    "build_relative_improvement_rows",
    "generate_full_report",
    "save_relative_improvement_table",
]


if __name__ == "__main__":
    from _analysis.cli import main

    main()

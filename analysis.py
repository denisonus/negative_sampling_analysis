"""Compatibility entry point for automatic experiment analysis."""

from _analysis.report import generate_full_report

__all__ = ["generate_full_report"]


if __name__ == "__main__":
    from _analysis.cli import main

    main()

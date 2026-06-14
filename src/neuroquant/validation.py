"""Input validation checks.

Validation is the first line of defence in any analytics workflow. These
functions fail fast with clear, actionable messages so that bad inputs are
caught before they silently corrupt downstream metrics.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("close",)


class ValidationError(ValueError):
    """Raised when an input fails a data-quality or configuration check."""


def validate_price_frame(
    frame: pd.DataFrame,
    min_rows: int = 2,
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """Validate a price/signal DataFrame and return it unchanged on success.

    Checks performed:
      * input is a non-empty pandas DataFrame
      * all required columns are present (e.g. ``close``)
      * required columns contain no missing / NaN values
      * required columns are strictly positive (a level series must be > 0)
      * there are enough rows for downstream rolling windows

    Parameters
    ----------
    frame:
        Candidate DataFrame to validate.
    min_rows:
        Minimum number of rows required.
    required_columns:
        Columns that must exist and be clean.

    Raises
    ------
    ValidationError
        If any check fails, with a message explaining the problem.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            f"Expected a pandas DataFrame, got {type(frame).__name__}."
        )

    if frame.empty:
        raise ValidationError("Input frame is empty; no rows to analyse.")

    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise ValidationError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(frame.columns)}."
        )

    if len(frame) < min_rows:
        raise ValidationError(
            f"Insufficient rows: need at least {min_rows}, got {len(frame)}. "
            "Rolling windows cannot be computed on too few observations."
        )

    for col in required_columns:
        series = frame[col]
        if series.isna().any():
            n_missing = int(series.isna().sum())
            raise ValidationError(
                f"Column '{col}' contains {n_missing} missing/NaN value(s). "
                "Fill or drop them before running the analysis."
            )
        if (series <= 0).any():
            n_bad = int((series <= 0).sum())
            raise ValidationError(
                f"Column '{col}' contains {n_bad} non-positive value(s). "
                "A level series must be strictly positive."
            )

    return frame


def validate_window_config(short_window: int, long_window: int) -> None:
    """Validate moving-average window configuration.

    Ensures both windows are positive integers and that the short window is
    strictly smaller than the long window (otherwise the crossover signal is
    meaningless).

    Raises
    ------
    ValidationError
        If the configuration is invalid.
    """
    if not isinstance(short_window, int) or not isinstance(long_window, int):
        raise ValidationError("Window sizes must be integers.")

    if short_window <= 0 or long_window <= 0:
        raise ValidationError(
            f"Window sizes must be positive. "
            f"Got short_window={short_window}, long_window={long_window}."
        )

    if short_window >= long_window:
        raise ValidationError(
            f"short_window ({short_window}) must be strictly less than "
            f"long_window ({long_window}). A crossover needs a faster and a "
            "slower moving average."
        )

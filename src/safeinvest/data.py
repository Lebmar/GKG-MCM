"""Load complete market paths while explicitly gating final evaluation data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

ASSET_COLUMNS = ("equity_return", "bond_return", "bill_proxy_return")
REQUIRED_COLUMNS = ("scenario_id", "period", *ASSET_COLUMNS, "inflation")


def _scenario_ids(values: Iterable[int]) -> NDArray[np.int64]:
    requested = list(values)
    if not requested:
        raise ValueError("scenario_ids must contain at least one scenario")
    ids = []
    for value in requested:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("scenario IDs must be integers in 1..180")
        try:
            integer = int(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("scenario IDs must be integers in 1..180") from exc
        if integer != value or not 1 <= integer <= 180:
            raise ValueError("scenario IDs must be integers in 1..180")
        ids.append(integer)
    if len(set(ids)) != len(ids):
        raise ValueError("scenario_ids must not contain duplicates")
    return np.asarray(ids, dtype=np.int64)


def load_paths(
    path: str | Path,
    scenario_ids: Iterable[int],
    allow_final: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Return returns (n, 30, 3), inflation (n, 30), and ordered scenario IDs.

    IDs 121..180 cannot be requested unless ``allow_final=True`` is explicitly
    supplied. The gate runs before opening the CSV. Unrequested return values
    are never parsed or validated, avoiding accidental inspection of the final
    paths during development. Every selected path must contain exactly years
    1..30, once each, with finite returns and strictly positive gross factors.
    Input scenario order is preserved, while periods are sorted numerically.
    """
    ids = _scenario_ids(scenario_ids)
    if not isinstance(allow_final, bool):
        raise ValueError("allow_final must be explicitly True or False")
    if np.any(ids >= 121) and not allow_final:
        raise ValueError("final evaluation scenarios 121..180 require allow_final=True")
    selected: dict[int, dict[int, tuple[float, ...]]] = {int(sid): {} for sid in ids}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                sid = int(row["scenario_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid scenario ID at CSV row {row_number}") from exc
            if sid not in selected:
                continue
            try:
                period = int(row["period"])
                values = tuple(float(row[column]) for column in (*ASSET_COLUMNS, "inflation"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid market value at CSV row {row_number}") from exc
            if not 1 <= period <= 30:
                raise ValueError(f"scenario {sid} has period outside 1..30: {period}")
            if period in selected[sid]:
                raise ValueError(f"duplicate scenario-period key: ({sid}, {period})")
            if not all(np.isfinite(value) and value > -1.0 for value in values):
                raise ValueError(f"scenario {sid}, period {period}: rates must be finite with positive gross factors")
            selected[sid][period] = values
    expected = set(range(1, 31))
    for sid, periods in selected.items():
        if set(periods) != expected:
            missing_periods = sorted(expected - set(periods))
            raise ValueError(f"scenario {sid} is incomplete; missing periods: {missing_periods}")
    values = np.asarray([[selected[int(sid)][period] for period in range(1, 31)] for sid in ids])
    return values[:, :, :3], values[:, :, 3], ids

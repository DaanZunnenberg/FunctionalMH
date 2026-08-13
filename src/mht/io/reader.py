"""
CSV reader for pre-computed simulation results.

The :class:`Reader` class loads local CSV files from the ``simulations/``
directory.

NOTE: CLAUDE.md's "Data Storage Constraints" section states that this
framework must not read, parse, or process .csv/.xlsx/tabular files.
This module contradicts that constraint (it exists specifically to read
CSVs via pandas.read_csv). It is currently unreferenced by the rest of
``src/mht`` and unused in tests -- flagged here rather than silently
removed or "fixed", since resolving the contradiction is a product
decision, not an engineering one.
"""
from __future__ import annotations

from itertools import accumulate
from pathlib import Path
from typing import MutableMapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

__all__: list[str] = ["Reader"]

_DEFAULT_SIM_DIR: Path = Path(__file__).parent.parent.parent.parent / "simulations"


class Reader:
    """
    Read simulation CSV files into a dict of DataFrames.

    Parameters
    ----------
    files : list[str | Path]
        File names (bare name or full path) to load.
    index_col : str
        Column name to use as the DataFrame index (default: "Unnamed: 0").
    sim_dir : Path or str, optional
        Directory containing the simulation files.  Defaults to
        ``simulations/`` at the repository root.
    """

    def __init__(
        self,
        files: list[str | Path] | None = None,
        index_col: str = "Unnamed: 0",
        sim_dir: Path | str | None = None,
    ) -> None:
        self.sim_dir = Path(sim_dir) if sim_dir is not None else _DEFAULT_SIM_DIR
        self.index_col = index_col
        self.parsed_files: MutableMapping[str, pd.DataFrame] = {}

        if files is not None:
            self._files = [self._resolve(f) for f in files]
            self._read()

    def _resolve(self, f: str | Path) -> Path:
        """Return an absolute Path, stripping any leading slash or backslash."""
        name = Path(str(f).lstrip("/\\")).name
        return self.sim_dir / name

    def _read(self) -> None:
        for path in tqdm(self._files):
            if not path.is_file():
                raise FileNotFoundError(f"Simulation file not found: {path}")
            self.parsed_files[path.name] = pd.read_csv(
                path, index_col=self.index_col
            )

    @staticmethod
    def running_maximum(X: npt.ArrayLike) -> list[float]:
        """Cumulative running maximum of ``|X|``."""
        return list(accumulate(np.abs(X), max))

    def __repr__(self) -> str:
        return str(list(self.parsed_files.keys()))

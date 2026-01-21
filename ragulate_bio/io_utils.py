"""
Input and output helper functions.

These functions handle loading the gold-standard data and
subsetting it for experiments.  The stratified sampling logic
mirrors the notebook implementation.  Additional helpers could
read or write other artefacts (e.g. saving row outputs or metrics).
"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from . import config


def maybe_load_gold(path: Path = config.GOLD_PATH) -> Optional[pd.DataFrame]:
    """Load the gold-edge CSV if it exists; otherwise return ``None``."""
    if not path.exists():
        print(f"[warn] {path} not found.")
        return None
    return pd.read_csv(path)


def subset_gold(df: pd.DataFrame, n_rows: Optional[int], stratified: bool = True, seed: int = 42) -> pd.DataFrame:
    """Return a subset of the gold dataframe with optional stratification.

    Parameters
    ----------
    df:
        The full gold-standard DataFrame.
    n_rows:
        Number of rows to sample; if ``None`` or greater than the
        length of ``df``, the full dataframe is returned.
    stratified:
        Whether to sample equally from each label class (if a
        ``label`` column exists).
    seed:
        Random seed for reproducibility.
    """
    df = df.reset_index(drop=True)
    if n_rows is None or n_rows >= len(df):
        return df
    if stratified and 'label' in df.columns:
        rng = np.random.default_rng(seed)
        groups = list(df.groupby('label', sort=False))
        n_labels = len(groups)
        if n_labels == 0:
            return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
        base = n_rows // n_labels
        rem = n_rows - base * n_labels
        sizes = np.array([len(g) for _, g in groups], dtype=float)
        weights = sizes / sizes.sum() if sizes.sum() > 0 else np.ones_like(sizes) / len(sizes)
        extra_idx = set(rng.choice(len(groups), size=min(rem, len(groups)), replace=False, p=weights))
        parts: List[pd.DataFrame] = []
        caps: List[int] = []
        for i, (lab, g) in enumerate(groups):
            target = base + (1 if i in extra_idx else 0)
            k = min(target, len(g))
            if k > 0:
                parts.append(g.sample(n=k, random_state=seed))
            caps.append(len(g) - k)
        out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
        short = n_rows - len(out)
        if short > 0:
            pool_parts: List[pd.DataFrame] = []
            for (lab, g), cap, _taken in zip(groups, caps, [len(p) if isinstance(p, pd.DataFrame) else 0 for p in parts + [None] * (len(groups) - len(parts))]):
                if cap > 0:
                    if parts:
                        sel_idx = set(out.index[out['label'] == lab]) if 'label' in out.columns else set()
                        g_pool = g[~g.index.isin(sel_idx)]
                    else:
                        g_pool = g
                    if len(g_pool) > 0:
                        pool_parts.append(g_pool)
            pool = pd.concat(pool_parts, axis=0) if pool_parts else df.iloc[0:0].copy()
            if len(pool) > 0:
                add = pool.sample(n=min(short, len(pool)), random_state=seed)
                out = pd.concat([out, add], ignore_index=True)
        if len(out) > n_rows:
            out = out.sample(n=n_rows, random_state=seed)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)


__all__ = ["maybe_load_gold", "subset_gold"]
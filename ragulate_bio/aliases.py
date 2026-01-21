"""
Gene / TF alias utilities based on the HGNC complete set.

This module provides a small, cached interface around the HGNC
"complete set" file from genenames.org. It is intended for use
in query expansion (e.g. adding gene/T F synonyms to retrieval
queries in RAGulate).

Expected input:
    - A tab-separated HGNC file with at least the columns:
        * "symbol"        : current HGNC-approved symbol
        * "alias_symbol"  : optional aliases, '|' separated
        * "prev_symbol"   : optional previous symbols, '|' separated

By default, the path is taken from config.HGNC_COMPLETE_SET_FILE,
which should point to e.g.:
    /home/mehrdad/Research/RAGulate/data/structured/hgnc_complete_set.txt
"""

from __future__ import annotations

from typing import Dict, Set, Optional, Iterable, List, Tuple

import os
from dataclasses import dataclass, field

import pandas as pd

from . import config


# ---------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------


@dataclass
class _AliasState:
    loaded: bool = False
    aliases_by_canonical: Dict[str, Set[str]] = field(default_factory=dict)
    canonical_by_alias: Dict[str, str] = field(default_factory=dict)
    source_path: Optional[str] = None


_STATE = _AliasState()


# ---------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------


def _norm_symbol(sym: str) -> str:
    """
    Normalise a gene/TF symbol for matching:
    - strip whitespace
    - uppercase
    """
    if sym is None:
        return ""
    return sym.strip().upper()


def _split_pipe_field(val: str) -> List[str]:
    """Split a '|'-separated field into a list of cleaned strings."""
    if not isinstance(val, str) or not val.strip():
        return []
    parts = [v.strip() for v in val.split("|")]
    return [p for p in parts if p]


# ---------------------------------------------------------------------
# Loading HGNC
# ---------------------------------------------------------------------


def _load_hgnc_table(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the HGNC complete set as a DataFrame.

    Parameters
    ----------
    path : Optional[str]
        Path to the HGNC TSV file. If None, uses config.HGNC_COMPLETE_SET_FILE.

    Returns
    -------
    pd.DataFrame
        HGNC table with all columns as strings.
    """
    if path is None:
        path = str(config.HGNC_COMPLETE_SET_FILE)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HGNC complete set file not found at {path}. "
            "Please download it from genenames.org and update config.HGNC_COMPLETE_SET_FILE."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        na_filter=False,
        low_memory=False,
    )
    return df


def _build_alias_maps_from_hgnc(df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Build alias maps from the HGNC DataFrame.

    Returns
    -------
    aliases_by_canonical : dict
        canonical_symbol -> set of all known symbols (canonical + aliases + previous).
    canonical_by_alias : dict
        alias_symbol -> canonical_symbol (first one wins if conflicts).
    """
    aliases_by_canonical: Dict[str, Set[str]] = {}
    canonical_by_alias: Dict[str, str] = {}

    # Some HGNC files use "symbol", "alias_symbol", "prev_symbol"
    # (we treat missing columns gracefully).
    has_alias = "alias_symbol" in df.columns
    has_prev = "prev_symbol" in df.columns

    for _, row in df.iterrows():
        base = _norm_symbol(row.get("symbol", ""))
        if not base:
            continue

        # Start alias set with the canonical symbol itself
        ali_set = aliases_by_canonical.setdefault(base, set())
        ali_set.add(base)

        # Collect aliases from alias_symbol and prev_symbol
        alias_candidates: List[str] = []
        if has_alias:
            alias_candidates.extend(_split_pipe_field(row["alias_symbol"]))
        if has_prev:
            alias_candidates.extend(_split_pipe_field(row["prev_symbol"]))

        for raw_alias in alias_candidates:
            alias = _norm_symbol(raw_alias)
            if not alias or alias == base:
                continue

            ali_set.add(alias)

            # Only assign canonical if this alias hasn't been seen before.
            # If conflicts exist across genes, we keep the first seen
            # to avoid hard failures; they are rare and mostly in
            # ambiguous symbols anyway.
            if alias not in canonical_by_alias:
                canonical_by_alias[alias] = base

    return aliases_by_canonical, canonical_by_alias


def ensure_aliases_loaded(path: Optional[str] = None, verbose: bool = True) -> None:
    """
    Ensure that HGNC aliases are loaded into memory.

    This function is idempotent; subsequent calls are no-ops unless
    a different `path` is provided (in which case it reloads).
    """
    global _STATE

    eff_path = path or str(config.HGNC_COMPLETE_SET_FILE)

    # Reload if not loaded yet, or if path changes.
    if _STATE.loaded and _STATE.source_path == eff_path:
        return

    if verbose and config.VERBOSE >= 1:
        print(f"[aliases] Loading HGNC aliases from {eff_path}")

    df = _load_hgnc_table(eff_path)
    aliases_by_canonical, canonical_by_alias = _build_alias_maps_from_hgnc(df)

    _STATE.aliases_by_canonical = aliases_by_canonical
    _STATE.canonical_by_alias = canonical_by_alias
    _STATE.loaded = True
    _STATE.source_path = eff_path

    if verbose and config.VERBOSE >= 1:
        n_canon = len(aliases_by_canonical)
        n_alias = len(canonical_by_alias)
        print(f"[aliases] Loaded {n_canon} canonical symbols with {n_alias} alias mappings")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def get_canonical_symbol(sym: str) -> str:
    """
    Return the canonical HGNC symbol for a given symbol/alias.

    If the symbol is unknown, returns the normalised input itself.
    """
    ensure_aliases_loaded(verbose=False)
    s = _norm_symbol(sym)
    if not s:
        return ""

    # If s itself is canonical, keep it; otherwise map via alias
    if s in _STATE.aliases_by_canonical:
        return s
    return _STATE.canonical_by_alias.get(s, s)


def get_all_aliases(sym: str, include_self: bool = True) -> List[str]:
    """
    Return all known aliases for the (possibly non-canonical) symbol.

    Parameters
    ----------
    sym : str
        Input symbol or alias.
    include_self : bool
        If True, ensures the (normalised) input symbol is present.

    Returns
    -------
    List[str]
        Sorted list of aliases (uppercased symbols).
    """
    ensure_aliases_loaded(verbose=False)
    if not sym:
        return []

    canonical = get_canonical_symbol(sym)
    ali_set = set(_STATE.aliases_by_canonical.get(canonical, set()))

    if include_self:
        ali_set.add(_norm_symbol(sym))

    # Deterministic ordering for reproducibility
    return sorted(ali_set)


def expand_symbol_for_query(
    sym: str,
    include_self: bool = True,
    max_aliases: Optional[int] = 5,
) -> List[str]:
    """
    Expand a symbol into a small list of symbols for query expansion.

    Example:
        TP53 -> ["TP53", "LFS1", "P53"]   (order deterministic, canonical first)
    """
    ensure_aliases_loaded(verbose=False)

    if not sym:
        return []

    base = _norm_symbol(sym)
    canonical = get_canonical_symbol(base)

    # All aliases stored for this canonical symbol (incl. canonical itself)
    ali_set = set(_STATE.aliases_by_canonical.get(canonical, set()))

    # We will make `base` the first element explicitly
    ali_set.discard(base)

    # Deterministic ordering of the remaining aliases
    alias_list = sorted(ali_set)

    if max_aliases is not None and max_aliases >= 0:
        alias_list = alias_list[:max_aliases]

    if include_self:
        return [base] + alias_list
    return alias_list




def describe_symbol(sym: str) -> Dict[str, List[str] | str]:
    """
    Convenience helper for debugging / inspection.

    Returns a small dictionary describing:
      - canonical symbol
      - all aliases
    """
    canonical = get_canonical_symbol(sym)
    aliases = get_all_aliases(sym, include_self=True)
    return {
        "input": _norm_symbol(sym),
        "canonical": canonical,
        "aliases": aliases,
    }


__all__ = [
    "ensure_aliases_loaded",
    "get_canonical_symbol",
    "get_all_aliases",
    "expand_symbol_for_query",
    "describe_symbol",
]

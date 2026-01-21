# gold_standard.py
"""
Gold standard construction for RAGulate.

This module builds an evaluation-ready gold standard from CollectRI-style
documents. It produces a DataFrame with columns:

    tf, target, context, label, pmids

and can optionally write this to CSV. A convenience helper
`ensure_gold_standard_v2` will build the file only if it does not
already exist, and otherwise load it.

Typical usage from a notebook:

    from modules.gold_standard import ensure_gold_standard_v2
    from modules import config

    gold_df = ensure_gold_standard_v2(
        path_csv=config.GOLD_PATH,   # ../outputs/ragulate_gold_standard_v2.csv
        ctx_thresh=0.30,
        n_pos_per_ctx=200,
        n_neg_per_ctx=200,
        n_pos_total=1000,           # global caps
        n_neg_total=1000,
    )

If `collectri_docs` is not provided, the module will load it from
`config.COLLECTRI_DOCS_PATH` (a pickle file).
"""

import csv
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any, Optional, Union
import hashlib
import numpy as np
import pandas as pd
from . import config


# -------------------------
# Default hyperparameters
# -------------------------

# Use config.SEED for consistency across the project
RNG_SEED = getattr(config, "SEED", 42)
CTX_THRESH = 0.30

# Reuse config.MAX_PMIDS_PER_EDGE if available; otherwise default to 5
MAX_PMIDS_PER_EDGE = getattr(config, "MAX_PMIDS_PER_EDGE", 5)

TOP_K_RETRIEVE = 15  # not directly used here, but kept for reference

# Per-context sampling (we will also cap by global totals)
N_POS_PER_CTX = 200   # you can tune these
N_NEG_PER_CTX = 200

# Global caps for total positives/negatives in the final gold standard
N_POS_TOTAL_DEFAULT = 1000
N_NEG_TOTAL_DEFAULT = 1000


# -------------------------
# Basic utilities
# -------------------------

def _meta(d: Any) -> Dict[str, Any]:
    """
    Extract metadata from a CollectRI-style document.

    Supports multiple shapes:

    - LlamaIndex-style Document objects with a `.metadata` dict
    - Objects with a `.meta` dict
    - Plain dicts with either:
        * top-level fields, or
        * a nested 'meta' dict

    Always returns a dict; falls back to {} if nothing usable is found.
    """
    # 1) LlamaIndex / pydantic Document style: d.metadata is a dict
    if hasattr(d, "metadata") and isinstance(getattr(d, "metadata"), dict):
        return getattr(d, "metadata")

    # 2) Objects with a `.meta` attribute
    if hasattr(d, "meta") and isinstance(getattr(d, "meta"), dict):
        return getattr(d, "meta")

    # 3) Plain dicts
    if isinstance(d, dict):
        # If there's an inner 'meta' dict, prefer that
        if "meta" in d and isinstance(d["meta"], dict):
            return d["meta"]
        return d

    # 4) Fallback: nothing usable
    return {}



def _norm(x: Any) -> str:
    """Normalise TF / target names to a canonical lowercase form."""
    if x is None:
        return ""
    return str(x).strip().lower()


def _pmids_list(x: Any) -> List[str]:
    """
    Flexibly parse a PMID field into a list of strings.
    Handles single int/str, list-like, or ';'-separated strings.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(p).strip() for p in x if str(p).strip()]
    s = str(x).strip()
    if not s:
        return []
    # Try semicolon-separated first
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    # Otherwise assume single PMID
    return [s]


# -------------------------
# Build context → (tf,target) → pmids
# -------------------------

def build_ctx2edges(
    docs: List[Dict[str, Any]],
    ctx_thresh: float = CTX_THRESH,
    min_pmids_per_edge: int = 1,
) -> Dict[str, Dict[Tuple[str, str], Set[str]]]:
    """
    Build a mapping:
        ctx2edges[context][(tf, target)] = set(pmids)

    Uses contextual scores to keep only edges with context score >= ctx_thresh.
    Edges with fewer than min_pmids_per_edge PMIDs are discarded.
    """
    ctx2edges: Dict[str, Dict[Tuple[str, str], Set[str]]] = defaultdict(lambda: defaultdict(set))

    for d in docs:
        m = _meta(d)
        tf = _norm(m.get("TF"))
        tg = _norm(m.get("Target"))
        if not tf or not tg:
            continue

        ctx = (m.get("Context") or "general").lower().strip()

        # Context score logic: use ContextScores[ctx] if present,
        # otherwise fall back to ContextScore.
        cs = m.get("ContextScores") or {}
        score = float(cs.get(ctx, m.get("ContextScore", 0.0) or 0.0))
        if score < ctx_thresh:
            continue

        pmids = _pmids_list(m.get("PMID"))
        if len(pmids) < min_pmids_per_edge:
            continue

        for p in pmids:
            ctx2edges[ctx][(tf, tg)].add(p)

    return ctx2edges


# -------------------------
# Positive / negative sampling
# -------------------------

def _sample_pos(
    ctx2edges: Dict[str, Dict[Tuple[str, str], Set[str]]],
    contexts: List[str],
    n_per_ctx: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sample positive edges (label=1) per context.

    Each row includes the full set of PMIDs for that edge (we later
    subsample for the final CSV).
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    for ctx in contexts:
        edge_dict = ctx2edges.get(ctx, {})
        pool = sorted(edge_dict.keys())
        if not pool:
            continue
        pick = pool if len(pool) <= n_per_ctx else rng.sample(pool, n_per_ctx)
        for tf, tg in pick:
            pmids = sorted(edge_dict[(tf, tg)])
            rows.append(
                {
                    "tf": tf,
                    "target": tg,
                    "context": ctx,
                    "label": 1,
                    "pmids": pmids,  # full set; subsampled later
                }
            )
    return rows


def _sample_neg(
    ctx2edges: Dict[str, Dict[Tuple[str, str], Set[str]]],
    contexts: List[str],
    n_per_ctx: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Context-shuffled negatives: choose TF–target edges that are real in
    *some* context but NOT in the current context.

    Negatives have label=0 and pmids=[] (no gold evidence for that
    (tf, target, context) combination).
    """
    rng = random.Random(seed)
    by_ctx = {ctx: set(ctx2edges.get(ctx, {}).keys()) for ctx in contexts}
    all_edges = set().union(*by_ctx.values()) if by_ctx else set()

    rows: List[Dict[str, Any]] = []
    for ctx in contexts:
        pos_edges_here = by_ctx.get(ctx, set())
        neg_pool = sorted(all_edges - pos_edges_here)
        if not neg_pool:
            continue
        pick = neg_pool if len(neg_pool) <= n_per_ctx else rng.sample(neg_pool, n_per_ctx)
        for tf, tg in pick:
            rows.append(
                {
                    "tf": tf,
                    "target": tg,
                    "context": ctx,
                    "label": 0,
                    "pmids": [],  # no gold pmids for negatives (by design)
                }
            )
    return rows

def _subsample_pmids_for_row(
    row: Dict[str, Any],
    k: int = MAX_PMIDS_PER_EDGE,
    seed: int = RNG_SEED,
) -> List[str]:
    """
    Deterministic subsampling of PMIDs per positive edge.

    Uses (tf, target, context) to derive a stable, reproducible seed
    instead of Python's salted hash, so that the same (tf, tg, ctx)
    triple always yields the same subset of PMIDs across runs.
    """
    pmids = row.get("pmids", []) or []
    if not pmids:
        return []

    tf = row["tf"]
    tg = row["target"]
    ctx = row["context"]

    # Stable hash of (tf, tg, ctx)
    key = f"{tf}|{tg}|{ctx}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % (2**32)

    rng = random.Random(seed + h)

    if len(pmids) <= k:
        return list(pmids)
    return rng.sample(pmids, k)




# -------------------------
# Main constructors
# -------------------------

def make_gold_standard_v2(
    collectri_docs: List[Dict[str, Any]],
    ctx_thresh: float = CTX_THRESH,
    min_pmids_per_edge: int = 1,
    n_pos_per_ctx: int = N_POS_PER_CTX,
    n_neg_per_ctx: int = N_NEG_PER_CTX,
    n_pos_total: Optional[int] = N_POS_TOTAL_DEFAULT,
    n_neg_total: Optional[int] = N_NEG_TOTAL_DEFAULT,
    max_pmids_per_edge: int = MAX_PMIDS_PER_EDGE,
    seed: int = RNG_SEED,
    context_whitelist: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build an improved gold standard with columns:
        tf, target, context, label, pmids

    - Positives are edges with context score >= ctx_thresh and at least
      min_pmids_per_edge PMIDs in that context.
    - Negatives are context-shuffled (edge real somewhere else but not
      in this context).
    - For positives, pmids is subsampled to at most max_pmids_per_edge.
    - For negatives, pmids=[].
    - We *approximate* n_pos_total / n_neg_total by sampling from the
      per-context pools if there are more rows than those totals.
    """
    # 1) Build context → edges → pmids
    ctx2edges = build_ctx2edges(
        collectri_docs,
        ctx_thresh=ctx_thresh,
        min_pmids_per_edge=min_pmids_per_edge,
    )
    all_contexts = sorted(ctx2edges.keys())
    if context_whitelist:
        contexts = [c for c in all_contexts if c in context_whitelist]
    else:
        contexts = all_contexts

    print(f"[gold] contexts available: {len(all_contexts)}")
    print(f"[gold] contexts used:      {len(contexts)}")

    # 2) Sample positives and negatives per context
    pos_rows = _sample_pos(ctx2edges, contexts, n_pos_per_ctx, seed)
    neg_rows = _sample_neg(ctx2edges, contexts, n_neg_per_ctx, seed)

    print(f"[gold] sampled positives (before global cap): {len(pos_rows)}")
    print(f"[gold] sampled negatives (before global cap): {len(neg_rows)}")

    # 3) Apply global caps if requested
    rng = random.Random(seed)
    if n_pos_total is not None and len(pos_rows) > n_pos_total:
        pos_rows = rng.sample(pos_rows, n_pos_total)
        print(f"[gold] capped positives to: {len(pos_rows)} (target {n_pos_total})")

    if n_neg_total is not None and len(neg_rows) > n_neg_total:
        neg_rows = rng.sample(neg_rows, n_neg_total)
        print(f"[gold] capped negatives to: {len(neg_rows)} (target {n_neg_total})")

    rows = pos_rows + neg_rows

    # 4) Subsample PMIDs for positives, leave negatives empty
    for r in rows:
        if r["label"] == 1:
            r["pmids"] = _subsample_pmids_for_row(
                r,
                k=max_pmids_per_edge,
                seed=seed,
            )
        else:
            r["pmids"] = []

    # 5) Build DataFrame; pmids is kept as a Python list here
    gold_df = pd.DataFrame(rows, columns=["tf", "target", "context", "label", "pmids"])

    # Shuffle rows for good measure (optional)
    gold_df = gold_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[gold] final gold standard shape: {gold_df.shape}")
    print("[gold] label counts:")
    print(gold_df["label"].value_counts())

    return gold_df


def save_gold_standard_v2(
    gold_df: pd.DataFrame,
    path_csv: Union[str, Path] = "gold_standard_v2.csv",
) -> None:
    """
    Save the gold DataFrame to CSV; pmids is stored as a JSON-like list string.
    Your existing _maybe_eval_pmids_field in the pipeline can parse this.
    """
    path_csv = str(path_csv)
    df = gold_df.copy()
    df["pmids"] = df["pmids"].apply(lambda x: json.dumps(x))
    # Ensure parent directory exists
    out_path = Path(path_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[gold] wrote {out_path} with {len(df)} rows")


# -------------------------
# Convenience entry point
# -------------------------

def ensure_gold_standard_v2(
    collectri_docs: Optional[List[Dict[str, Any]]] = None,
    path_csv: Union[str, Path] = None,
    ctx_thresh: float = CTX_THRESH,
    min_pmids_per_edge: int = 1,
    n_pos_per_ctx: int = N_POS_PER_CTX,
    n_neg_per_ctx: int = N_NEG_PER_CTX,
    n_pos_total: Optional[int] = N_POS_TOTAL_DEFAULT,
    n_neg_total: Optional[int] = N_NEG_TOTAL_DEFAULT,
    max_pmids_per_edge: int = MAX_PMIDS_PER_EDGE,
    seed: int = RNG_SEED,
    context_whitelist: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build the v2 gold standard *only if* the CSV does not exist.

    If `path_csv` exists:
        - Load it into a DataFrame.
        - Parse the `pmids` column back into Python lists.

    If it does not exist:
        - If `collectri_docs` is None, load it from config.COLLECTRI_DOCS_PATH.
        - Construct the gold standard via `make_gold_standard_v2`.
        - Save it via `save_gold_standard_v2`.
        - Return the newly built DataFrame.
    """
    # Default path: config.GOLD_PATH
    if path_csv is None:
        path_csv = config.GOLD_PATH
    path_csv = Path(path_csv)

    if path_csv.exists():
        print(f"[gold] found existing file: {path_csv}, loading")
        df = pd.read_csv(path_csv)

        def _parse_pmids_cell(x: Any) -> List[str]:
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                x = x.strip()
                if not x:
                    return []
                try:
                    val = json.loads(x)
                    if isinstance(val, list):
                        return [str(p) for p in val]
                except Exception:
                    # Fallback: treat as single PMID string
                    return [x]
            return []

        if "pmids" in df.columns:
            df["pmids"] = df["pmids"].apply(_parse_pmids_cell)
        return df

    # If file does not exist, build and save
    print(f"[gold] no existing gold file at {path_csv}, building a new one")

    # Load CollectRI docs if not provided
    if collectri_docs is None:
        coll_path = config.COLLECTRI_DOCS_PATH
        print(f"[gold] loading CollectRI docs from: {coll_path}")
        with open(coll_path, "rb") as f:
            collectri_docs = pickle.load(f)
        print(f"[gold] loaded {len(collectri_docs)} CollectRI documents")

    gold_df = make_gold_standard_v2(
        collectri_docs=collectri_docs,
        ctx_thresh=ctx_thresh,
        min_pmids_per_edge=min_pmids_per_edge,
        n_pos_per_ctx=n_pos_per_ctx,
        n_neg_per_ctx=n_neg_per_ctx,
        n_pos_total=n_pos_total,
        n_neg_total=n_neg_total,
        max_pmids_per_edge=max_pmids_per_edge,
        seed=seed,
        context_whitelist=context_whitelist,
    )
    save_gold_standard_v2(gold_df, path_csv=path_csv)
    return gold_df


__all__ = [
    "RNG_SEED",
    "CTX_THRESH",
    "MAX_PMIDS_PER_EDGE",
    "TOP_K_RETRIEVE",
    "N_POS_PER_CTX",
    "N_NEG_PER_CTX",
    "N_POS_TOTAL_DEFAULT",
    "N_NEG_TOTAL_DEFAULT",
    "build_ctx2edges",
    "make_gold_standard_v2",
    "save_gold_standard_v2",
    "ensure_gold_standard_v2",
]

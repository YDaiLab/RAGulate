"""
Metric computation utilities.

This module exposes functions to compute AUROC, AUPRC, optimal F1
and to perform permutation tests on binary classification scores.
These helpers wrap ``scikit-learn`` functions when available and
fall back gracefully otherwise.
"""

from typing import List, Optional, Dict, Any, Sequence

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_recall_curve,
    )
except Exception:
    roc_auc_score = None  # type: ignore
    average_precision_score = None  # type: ignore
    f1_score = None  # type: ignore
    precision_recall_curve = None  # type: ignore

from . import config


def recall_at_k(gold_set: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Recall@K for a single edge."""
    g = {x for x in gold_set if x}
    if not g:
        return 0.0
    r = set(retrieved[:k])
    return len(g & r) / len(g)


def precision_at_k(gold_set: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Precision@K for a single edge."""
    r = retrieved[:k]
    if not r:
        return 0.0
    g = {x for x in gold_set if x}
    return len(g & set(r)) / len(r)

def average_precision_for_edge(
    gold_set: Sequence[str],
    retrieved: Sequence[str],
) -> float:
    """
    Compute average precision (AP) for a single edge.

    Average precision is the mean of precisions computed at each
    position where a relevant document is retrieved. If there are no
    relevant documents, AP is 0.
    """
    g = {x for x in gold_set if x}
    if not g:
        return 0.0
    num_hits = 0
    precisions: List[float] = []
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in g:
            num_hits += 1
            precisions.append(num_hits / idx)
    return float(np.sum(precisions) / len(g)) if precisions else 0.0

    
def mrr_for_edge(gold_set: Sequence[str], retrieved: Sequence[str]) -> float:
    """Mean reciprocal rank for a single edge."""
    g = {x for x in gold_set if x}
    if not g:
        return 0.0
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in g:
            return 1.0 / idx
    return 0.0


def compute_retrieval_metrics_from_outputs(
    rows: List[Dict[str, Any]],
    ks: List[int] = [1, 3, 5, 10],
    gold_key: str = "gold_pmids",
    retrieved_key: str = "retrieved_pmids",
) -> Dict[str, float]:
    """
    Compute retrieval metrics across all edges:
      - recall@k, precision@k for each k in ks
      - mean reciprocal rank (mrr)
      - mean average precision (map)
    """
    if not rows:
        base = {f"recall@{k}": 0.0 for k in ks}
        base.update({f"precision@{k}": 0.0 for k in ks})
        base.update({"mrr": 0.0, "map": 0.0})
        return base

    recalls = {k: [] for k in ks}
    precs = {k: [] for k in ks}
    mrrs: List[float] = []
    aps: List[float] = []

    for r in rows:
        gold = r.get(gold_key, [])
        ret = r.get(retrieved_key, [])
        for k in ks:
            recalls[k].append(recall_at_k(gold, ret, k))
            precs[k].append(precision_at_k(gold, ret, k))
        mrrs.append(mrr_for_edge(gold, ret))
        aps.append(average_precision_for_edge(gold, ret))

    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        out[f"precision@{k}"] = float(np.mean(precs[k])) if precs[k] else 0.0
    out["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0
    out["map"] = float(np.mean(aps)) if aps else 0.0
    return out



def evaluate_binary(y_true: List[int], y_score: List[float]) -> Dict[str, Optional[float]]:
    """Compute AUROC, AUPRC and best F1 on binary predictions."""
    out: Dict[str, Optional[float]] = {
        "auroc": None,
        "auprc": None,
        "best_f1": None,
        "best_thr": None,
    }
    try:
        if roc_auc_score is not None:
            out["auroc"] = float(roc_auc_score(y_true, y_score))
        if average_precision_score is not None:
            out["auprc"] = float(average_precision_score(y_true, y_score))
        if precision_recall_curve is not None and f1_score is not None:
            prec, rec, thr = precision_recall_curve(y_true, y_score)
            f1s = 2 * (prec * rec) / np.clip(prec + rec, 1e-8, None)
            i = int(np.argmax(f1s))
            out["best_f1"] = float(f1s[i])
            out["best_thr"] = (
                float(thr[i - 1]) if i > 0 and i - 1 < len(thr) else 0.5
            )
    except Exception:
        pass
    return out


def classification_metrics(
    y_true: List[int], y_score: List[float], thr: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """
    Full binary classification metrics:

    - AUROC
    - AUPRC
    - best F1 and corresponding threshold (from PR curve)
    - accuracy at the chosen threshold
    """
    base = evaluate_binary(y_true, y_score)
    thr_eff = thr if thr is not None else base.get("best_thr")
    if thr_eff is None:
        thr_eff = 0.5

    y_true_arr = np.array(y_true, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    y_pred = (y_score_arr >= thr_eff).astype(int)
    acc = float((y_pred == y_true_arr).mean())

    base["accuracy"] = acc
    base["thr_used"] = float(thr_eff)
    return base


def compute_classification_metrics_from_outputs(
    rows: List[Dict[str, Any]],
    label_key: str = "label",
    score_key: str = "score",
) -> Dict[str, Any]:
    """
    Expect rows like:
      {
        'label': 0/1,
        'score': float in [0,1],
        ...
      }
    """
    if not rows:
        return {}
    y_true = np.array([r[label_key] for r in rows], dtype=int)
    y_score = np.array([r[score_key] for r in rows], dtype=float)
    return classification_metrics(list(y_true), list(y_score))


def permutation_test(
    y_true: List[int],
    y_score: List[float],
    n_perm: int = 100,
    metric: str = "auroc",
    seed: int = config.SEED,
) -> Optional[Dict[str, float]]:
    """Perform a permutation test on binary scores to estimate null distributions."""
    if n_perm <= 0:
        return None
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        m = evaluate_binary(list(y_perm), y_score)
        vals.append(m.get(metric))
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    obs = evaluate_binary(y_true, y_score).get(metric)
    if obs is None:
        return None
    arr = np.array(vals)
    p = float((arr >= obs).mean())
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"p_value": p, "null_ci": (float(lo), float(hi)), "n_perm": len(vals)}

def compute_retrieval_metrics_dataframe(
    rows: List[Dict[str, Any]],
    ks: Sequence[int] = (1, 3, 5, 10),
    gold_key: str = "gold_pmids",
    retrieved_key: str = "retrieved_pmids",
    label_key: str = "label",
    positive_only: bool = True,
    require_gold_pmids: bool = True,
) -> "pd.DataFrame":
    """
    Compute retrieval metrics over a set of edges, optionally restricted to
    positive edges with at least one gold PMID.

    Parameters
    ----------
    rows:
        List of per-edge dicts from `run_retrieval_experiment`, each containing
        at least:
          - gold_key (default 'gold_pmids'): list of gold PMIDs
          - retrieved_key (default 'retrieved_pmids'): list of retrieved PMIDs
          - label_key (default 'label'): 0/1 class label

    ks:
        Cutoffs for recall@k and precision@k.

    positive_only:
        If True (default), only include edges with label == 1.

    require_gold_pmids:
        If True (default), only include edges where gold_key is non-empty.

    Returns
    -------
    A 1-row pandas DataFrame with:
      - recall@k, precision@k for each k in ks
      - mrr, map
      - n_queries: number of edges actually used in the metrics
    """
    import pandas as _pd

    if not rows:
        base = compute_retrieval_metrics_from_outputs([], ks=list(ks))
        base["n_queries"] = 0
        return _pd.DataFrame([base])

    # 1) Filter rows according to label / gold pmids
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        lbl = r.get(label_key, 1)  # default to 1 if missing
        gold = r.get(gold_key, []) or []

        if positive_only and lbl != 1:
            continue
        if require_gold_pmids and len(gold) == 0:
            continue

        filtered.append(r)

    if not filtered:
        base = compute_retrieval_metrics_from_outputs([], ks=list(ks))
        base["n_queries"] = 0
        return _pd.DataFrame([base])

    # 2) Compute retrieval metrics on the filtered subset
    metrics = compute_retrieval_metrics_from_outputs(
        filtered,
        ks=list(ks),
        gold_key=gold_key,
        retrieved_key=retrieved_key,
    )
    metrics["n_queries"] = len(filtered)

    return _pd.DataFrame([metrics])


__all__ = [
    "evaluate_binary",
    "classification_metrics",
    "compute_classification_metrics_from_outputs",
    "compute_retrieval_metrics_from_outputs",
    "permutation_test",
    "compute_retrieval_metrics_dataframe",
]

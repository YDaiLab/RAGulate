# modules/evidence.py

from typing import List, Dict, Any, Sequence, Set
import numpy as np


def _to_set(ids: Sequence[str]) -> Set[str]:
    return {i for i in ids if i}


def evaluate_pubmed_match(
    gold_ids_list: List[List[str]],
    pred_ids_list: List[List[str]],
    k: int = 5,
) -> Dict[str, float]:
    """
    Compute evidence-level metrics:
      - exact_match: fraction of edges with at least one predicted ID in gold
      - topk_match: same but restricted to top-k predictions
      - jaccard: mean Jaccard index between gold and predicted sets
    """
    assert len(gold_ids_list) == len(pred_ids_list)
    n = len(gold_ids_list)
    if n == 0:
        return {"exact_match": 0.0, "topk_match": 0.0, "mean_jaccard": 0.0}

    exact_hits = 0
    topk_hits = 0
    jaccards: List[float] = []

    for gold, pred in zip(gold_ids_list, pred_ids_list):
        gset = _to_set(gold)
        pset_full = _to_set(pred)
        pset_topk = _to_set(pred[:k]) if k > 0 else set()

        if gset & pset_full:
            exact_hits += 1
        if gset & pset_topk:
            topk_hits += 1

        if not gset and not pset_full:
            jaccards.append(1.0)
        else:
            inter = len(gset & pset_full)
            union = len(gset | pset_full)
            jaccards.append(0.0 if union == 0 else inter / union)

    return {
        "exact_match": exact_hits / n,
        "topk_match": topk_hits / n,
        "mean_jaccard": float(np.mean(jaccards)),
    }


def extract_pubmed_lists_from_rows(rows: List[Dict[str, Any]],
                                   gold_key: str = "gold_pmids",
                                   pred_key: str = "pred_pmids") -> Dict[str, Any]:
    """
    Helper for your row outputs (one dict per edge). Expects each row
    to have:
      - row[gold_key]: List[str]
      - row[pred_key]: List[str]
    """
    gold_ids = [r.get(gold_key, []) for r in rows]
    pred_ids = [r.get(pred_key, []) for r in rows]
    return {"gold_ids": gold_ids, "pred_ids": pred_ids}

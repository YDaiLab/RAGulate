from typing import List, Dict, Any
import pandas as pd

from . import retrieval, rag as rag_mod, config

def ragulate_score_edges(
    edges: pd.DataFrame,
    method_type: str = "hybrid",
    top_k: int = config.TOP_K_RETRIEVE,
) -> pd.DataFrame:
    """
    edges: DataFrame with columns 'tf','target','context'.
    Returns a DataFrame with support_score, retrieved_pmids, and rag_context.
    """
    rows: List[Dict[str, Any]] = []

    for _, row in edges.iterrows():
        tf = row["tf"]
        tgt = row["target"]
        ctx = row["context"]

        query = rag_mod.build_query_from_edge(tf, tgt, ctx)

        if method_type == "bm25":
            ret_docs = retrieval.retrieve_bm25(query, top_k=top_k)
        elif method_type == "ragulate":
            ret_docs = retrieval.retrieve_vector(query, top_k=top_k)
        elif method_type == "vanilla_rag":
            ret_docs = retrieval.retrieve_vanilla_rag(query, top_k=top_k)
        elif method_type == "hybrid":
            ret_docs = retrieval.retrieve_hybrid_bm25_vector(
                query,
                encoder_name=None,
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        pmids = [d.doc_id for d in ret_docs]
        scores = [float(d.score) for d in ret_docs]
        support = rag_mod.support_score_from_scores(scores, method="max")

        # build a compact RAG context (evidence snippets)
        # using the existing helper from rag.py
        ctx_text = rag_mod.build_rag_context(tf, tgt, ctx, ret_docs)

        rows.append(
            {
                "tf": tf,
                "target": tgt,
                "context": ctx,
                "retrieved_pmids": pmids,
                "retrieval_scores": scores,
                "support_score": support,
                "rag_context": ctx_text,
            }
        )

    return pd.DataFrame(rows)

# modules/experiments.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd

from . import config
from . import pipeline
from . import metrics as metric_utils
from . import evidence as evidence_utils


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    exp_id: str
    mode: str                      # "retrieval", "classification", "evidence"
    method_type: str               # "bm25", "ragulate", "vanilla_rag", "direct_llm", "hybrid", ...
    encoder_name: Optional[str]    # "biobert", "minilm", "default", None
    llm_name: Optional[str]        # HF model id or None
    top_k: int                     # retrieval top-k
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    notes: str = ""

    # --- retrieval-specific knobs (all defaulted so old calls still work) ---
    positives_only: bool = True            # restrict to label==1 edges
    query_mode: str = "tf_tg_ctx"          # "tf_tg_ctx" or "tf_tg_only"
    expand_context: bool = False           # use keyword expansion or not
    use_aliases: bool = False              # use HGNC alias expansion
    alias_max: int = 4                     # max aliases per symbol in query


def get_experiment_config(exp_id: str) -> ExperimentConfig:
    """
    Map an experiment ID (R1, L2, V3, D7, E3, etc.) to a concrete configuration.
    You can tweak this mapping to match your exact plan.
    """
    # ------------------------------------------------------------------
    # Retrieval-only experiments (Phase 1)
    # ------------------------------------------------------------------
    if exp_id == "R1":
        return ExperimentConfig(
            exp_id="R1",
            mode="retrieval",
            method_type="bm25",
            encoder_name=None,
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            notes="BM25 baseline (tf+target+context, positives only)",
        )
    if exp_id == "R2":
        return ExperimentConfig(
            exp_id="R2",
            mode="retrieval",
            method_type="ragulate",
            encoder_name="biobert",
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=False,
            notes="RAGulate BioBERT cosine (tf+target+context)",
        )
    if exp_id == "R3":
        return ExperimentConfig(
            exp_id="R3",
            mode="retrieval",
            method_type="ragulate",
            encoder_name="minilm",
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=False,
            notes="RAGulate MiniLM cosine (tf+target+context)",
        )
    if exp_id == "R4":
        return ExperimentConfig(
            exp_id="R4",
            mode="retrieval",
            method_type="ragulate",
            encoder_name="biobert",
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=True,
            mmr_lambda=0.7,
            notes="RAGulate BioBERT + MMR(0.7)",
        )
    if exp_id == "R5":
        return ExperimentConfig(
            exp_id="R5",
            mode="retrieval",
            method_type="ragulate",
            encoder_name="biobert",
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=True,
            mmr_lambda=0.5,
            notes="RAGulate BioBERT + MMR(0.5)",
        )
    if exp_id == "R6":
        return ExperimentConfig(
            exp_id="R6",
            mode="retrieval",
            method_type="vanilla_rag",
            encoder_name="default",
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            notes="Vanilla RAG default encoder",
        )
    if exp_id == "R7":
        return ExperimentConfig(
            exp_id="R7",
            mode="retrieval",
            method_type="ragulate",
            encoder_name="minilm",
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=True,
            mmr_lambda=0.7,
            notes="RAGulate MiniLM + MMR(0.7)",
        )
    if exp_id == "R8":
        return ExperimentConfig(
            exp_id="R8",
            mode="retrieval",
            method_type="hybrid",
            encoder_name=None,          # None -> default MiniLM in retrieve_hybrid_bm25_vector
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=False,
            mmr_lambda=0.0,
            notes="Hybrid BM25 + MiniLM",
            positives_only=True,
            query_mode="tf_tg_ctx",
            expand_context=False,
            use_aliases=False,
        )

    if exp_id == "R9":
        # This is the flagship retriever: Hybrid BM25 + MiniLM + HGNC aliases
        return ExperimentConfig(
            exp_id="R9",
            mode="retrieval",
            method_type="hybrid",
            encoder_name=None,          # None -> default MiniLM in retrieve_hybrid_bm25_vector
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=False,
            mmr_lambda=0.0,
            notes="Hybrid BM25 + MiniLM with HGNC alias expansion",
            positives_only=True,
            query_mode="tf_tg_ctx",
            expand_context=False,
            use_aliases=True,
            alias_max=4,
        )

    # ------------------------------------------------------------------
    # Level 1: RAGulate vs BM25
    # ------------------------------------------------------------------
    if exp_id == "L1":  # BM25 + simple decision
        return ExperimentConfig(
            exp_id="L1",
            mode="classification",
            method_type="bm25_simple",
            encoder_name=None,
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            notes="BM25 + heuristic/simple classifier",
        )
    if exp_id == "L2":  # BM25 + LLM
        return ExperimentConfig(
            exp_id="L2",
            mode="classification",
            method_type="bm25_llm",
            encoder_name=None,
            llm_name=config.MISTRAL_MODEL_NAME,   # or your chosen default LLM
            top_k=config.TOP_K_RETRIEVE,
            notes="BM25 + LLM",
        )

    if exp_id == "L3":  # RAGulate best config (now hybrid with alias expansion)
        return ExperimentConfig(
            exp_id="L3",
            mode="classification",
            method_type="hybrid",
            encoder_name=None,
            llm_name=config.MISTRAL_MODEL_NAME,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=False,
            notes="RAGulate flagship (hybrid BM25 + MiniLM with aliases)",
            use_aliases=True,
            alias_max=getattr(config, "MAX_ALIAS_PER_SYMBOL", 4),
        )

    # ------------------------------------------------------------------
    # Level 3: RAGulate vs vanilla RAG
    # ------------------------------------------------------------------
    if exp_id == "V1":
        return ExperimentConfig(
            exp_id="V1",
            mode="classification",
            method_type="ragulate",
            encoder_name="biobert",
            llm_name=config.MISTRAL_MODEL_NAME,  # example; pick your primary
            top_k=config.TOP_K_RETRIEVE,
            notes="RAGulate BioBERT",
        )
    if exp_id == "V2":
        return ExperimentConfig(
            exp_id="V2",
            mode="classification",
            method_type="ragulate",
            encoder_name="minilm",
            llm_name=config.MISTRAL_MODEL_NAME,
            top_k=config.TOP_K_RETRIEVE,
            notes="RAGulate MiniLM",
        )
    if exp_id == "V3":
        return ExperimentConfig(
            exp_id="V3",
            mode="classification",
            method_type="vanilla_rag",
            encoder_name="default",
            llm_name=config.MISTRAL_MODEL_NAME,
            top_k=config.TOP_K_RETRIEVE,
            notes="Vanilla RAG default encoder",
        )
    if exp_id == "V4":
        return ExperimentConfig(
            exp_id="V4",
            mode="classification",
            method_type="ragulate",
            encoder_name="biobert",
            llm_name=config.MISTRAL_MODEL_NAME,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=True,
            mmr_lambda=0.7,
            notes="RAGulate BioBERT + MMR",
        )
    if exp_id == "V5":
        return ExperimentConfig(
            exp_id="V5",
            mode="classification",
            method_type="ragulate",
            encoder_name="minilm",
            llm_name=config.MISTRAL_MODEL_NAME,
            top_k=config.TOP_K_RETRIEVE,
            use_mmr=True,
            mmr_lambda=0.7,
            notes="RAGulate MiniLM + MMR",
        )

    # ------------------------------------------------------------------
    # Level 2: Direct LLM vs RAGulate per LLM
    # ------------------------------------------------------------------
    if exp_id.startswith("D"):
        # we delegate to a helper that maps D1..D10 -> configs
        return _get_direct_llm_config(exp_id)

    # ------------------------------------------------------------------
    # Evidence faithfulness (E1–E5)
    # ------------------------------------------------------------------
    if exp_id.startswith("E"):
        return ExperimentConfig(
            exp_id=exp_id,
            mode="evidence",
            method_type="evidence",
            encoder_name=None,
            llm_name=None,
            top_k=config.TOP_K_RETRIEVE,
            notes="Evidence-level PubMed ID evaluation",
        )

    raise ValueError(f"Unknown experiment ID: {exp_id}")


def _get_direct_llm_config(exp_id: str) -> ExperimentConfig:
    """
    Map D1..D5 to direct LLMs and D6..D10 to RAGulate + LLM.
    """
    direct_map = {
        "D1": config.BIOGPT_MODEL_NAME,
        "D2": config.MISTRAL_MODEL_NAME,
        "D3": config.LLAMA31_MODEL_NAME,
        "D4": config.PHI3_MODEL_NAME,
        "D5": config.QWEN25_MODEL_NAME,
    }
    rag_map = {
        "D6": config.BIOGPT_MODEL_NAME,
        "D7": config.MISTRAL_MODEL_NAME,
        "D8": config.LLAMA31_MODEL_NAME,
        "D9": config.PHI3_MODEL_NAME,
        "D10": config.QWEN25_MODEL_NAME,
    }

    if exp_id in direct_map:
        return ExperimentConfig(
            exp_id=exp_id,
            mode="classification",
            method_type="direct_llm",
            encoder_name=None,
            llm_name=direct_map[exp_id],
            top_k=0,
            notes=f"Direct LLM {direct_map[exp_id]}",
        )

    if exp_id in rag_map:
        return ExperimentConfig(
            exp_id=exp_id,
            mode="classification",
            method_type="hybrid",
            encoder_name=None,
            llm_name=rag_map[exp_id],
            top_k=config.TOP_K_RETRIEVE,
            notes=f"RAGulate hybrid + {rag_map[exp_id]}",
        )

    raise ValueError(f"Unknown direct-LLM experiment ID: {exp_id}")


def run_experiment(exp_cfg: ExperimentConfig,
                   gold_df: pd.DataFrame) -> Dict[str, Any]:
    """
    High-level entry point that runs a single experiment and returns a dict with:
      - 'rows': list of per-edge outputs
      - 'metrics': summary metrics
    """
    # --------------------------------------------------------------
    # Retrieval-only mode
    # --------------------------------------------------------------
    if exp_cfg.mode == "retrieval":
        out = pipeline.run_retrieval_experiment(
            gold_df=gold_df,
            method_type=exp_cfg.method_type,
            encoder_name=exp_cfg.encoder_name,
            top_k=exp_cfg.top_k,
            use_mmr=exp_cfg.use_mmr,
            mmr_lambda=exp_cfg.mmr_lambda,
            positives_only=exp_cfg.positives_only,
            query_mode=exp_cfg.query_mode,
            expand_context=exp_cfg.expand_context,
            use_aliases=exp_cfg.use_aliases,
            alias_max=exp_cfg.alias_max,
        )

        # Retrieval metrics
        retrieval_metrics = metric_utils.compute_retrieval_metrics_from_outputs(
            out, ks=[1, 3, 5, 10, 20, 50, 100]
        )
        metrics: Dict[str, Any] = dict(retrieval_metrics)

        # Optional: classification metrics over support_score vs label
        # Only if:
        #   - we did NOT restrict to positives_only, and
        #   - labels contain at least two classes.
        if not exp_cfg.positives_only:
            y_true = [r.get("label") for r in out if "label" in r]
            y_true_clean = [y for y in y_true if y is not None]
            if len(set(y_true_clean)) >= 2:
                try:
                    cls_metrics = metric_utils.compute_classification_metrics_from_outputs(
                        out, label_key="label", score_key="support_score"
                    )
                except Exception:
                    cls_metrics = {}
                prefixed_cls_metrics = {f"support_{k}": v for k, v in cls_metrics.items()}
                metrics.update(prefixed_cls_metrics)

        return {"rows": out, "metrics": metrics}

    # --------------------------------------------------------------
    # Classification mode
    # --------------------------------------------------------------
    if exp_cfg.mode == "classification":
        out = pipeline.run_classification_experiment(
            gold_df=gold_df,
            method_type=exp_cfg.method_type,
            encoder_name=exp_cfg.encoder_name,
            llm_name=exp_cfg.llm_name,
            top_k=exp_cfg.top_k,
            use_mmr=exp_cfg.use_mmr,
            mmr_lambda=exp_cfg.mmr_lambda,
            use_aliases=exp_cfg.use_aliases,
            alias_max=exp_cfg.alias_max,
        )

        metrics = metric_utils.compute_classification_metrics_from_outputs(out)
        return {"rows": out, "metrics": metrics}

    # --------------------------------------------------------------
    # Evidence mode (not wired yet)
    # --------------------------------------------------------------
    if exp_cfg.mode == "evidence":
        raise NotImplementedError(
            "Evidence-level experiments (E1–E5) are run via evidence.evaluate_evidence_suite"
        )

    raise ValueError(f"Unsupported mode: {exp_cfg.mode}")


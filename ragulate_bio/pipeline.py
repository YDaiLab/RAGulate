"""
End-to-end benchmarking pipeline for RAGulate.

This module glues together retrieval, RAG context construction, language
model inference and metric computation.  The top-level functions
accept a pandas DataFrame of gold-standard edges and produce
per-method score arrays and summary metrics.  Running multiple
epochs is supported via ``run_benchmark_epochs``.
"""

from typing import List, Dict, Any, Optional, Tuple
import ast
import time
import math
import re
import numpy as np
import pandas as pd

from . import config
from .logging_utils import print_config_banner
from .retrieval import get_retriever_safe
from .embedding import _load_cached_pubmed_json, _concat_title_abs

from .llm_models import (
    get_biogpt,
    get_mistral,
    get_llama31,
    get_phi3,
    get_qwen25,
    _llm_generate,
    _llm_generate_batch,
)

from .rag import (
    build_prompt_with_budget,
    _get_rag_ctx_cached,
    _prompt_template,
    _binary_from_text,
)
from .metrics import evaluate_binary, permutation_test

# For the new experiment wrappers
from . import retrieval
from . import rag as rag_mod
from . import metrics as metric_utils
from . import aliases as alias_mod


# ---------- Helpers for the new experiment API ----------


def _score_yes_no_biogpt(
    tok,
    mdl,
    prompt: str,
    return_logit_diff: bool = False,
) -> float:
    """
    Use BioGPT as a binary classifier.

    If return_logit_diff is False (default), return P(yes | prompt) computed
    from the logits of the next token restricted to {" yes", " no"}.

    If return_logit_diff is True, return the raw logit margin:
        logit(" yes") - logit(" no")
    for the next-token distribution.
    """
    import torch

    # Tokenise the prompt
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)

    with torch.no_grad():
        out = mdl(**inputs)

    # Logits for the next token (last position)
    logits = out.logits[0, -1]  # shape [vocab_size]

    # Token IDs for " yes" and " no"
    id_yes = tok(" yes", add_special_tokens=False)["input_ids"][0]
    id_no = tok(" no", add_special_tokens=False)["input_ids"][0]

    logit_yes = logits[id_yes]
    logit_no = logits[id_no]
    logit_diff = float((logit_yes - logit_no).item())

    if return_logit_diff:
        # Direct margin: >0 means model prefers "yes", <0 prefers "no"
        return logit_diff

    # Otherwise keep old behaviour: probability of "yes"
    sel = torch.stack([logit_yes, logit_no])  # [2]
    probs = torch.softmax(sel, dim=0)
    prob_yes = float(probs[0].item())
    return prob_yes


def _answer_yes_no_single(tok, mdl, prompt: str) -> Tuple[float, int, str]:
    """
    Run a single yes/no prompt.

    For BioGPT we use a logits-based classifier.
    For other models we use generation + text parsing.
    """
    if tok is None or mdl is None:
        raise ValueError("LLM not initialised")

    # Detect BioGPT by the underlying HF name
    name = getattr(getattr(mdl, "config", None), "_name_or_path", "")
    is_biogpt = config.BIOGPT_MODEL_NAME in str(name)

    if is_biogpt:
        score = _score_yes_no_biogpt(tok, mdl, prompt)
        pred_label = int(round(score))
        raw = f"biogpt_score={score:.3f}"
        return score, pred_label, raw

    # Default: generative + text parser
    txts = _llm_generate_batch(tok, mdl, [prompt], max_new_tokens=config.GEN_MAX_NEW)
    raw = txts[0] if txts else ""
    score = float(_binary_from_text(raw))
    pred_label = int(round(score))
    return score, pred_label, raw

# -------------------------------------------------------------------
# Helper to compute a continuous edge-level score
# -------------------------------------------------------------------

def _combine_llm_and_retrieval_scores(llm_score: float, retrieval_scores: List[float]) -> Tuple[float, float, float]:
    """
    Combine the LLM-derived probability with retrieval support into a unified
    continuous score for ranking edges.

    Parameters
    ----------
    llm_score : float
        The probability (in [0,1]) returned by the LLM that a regulatory
        relationship holds.
    retrieval_scores : list of floats
        Raw retrieval scores for the documents returned by the retriever.  If
        empty, retrieval support is treated as zero.

    Returns
    -------
    final_score : float
        A combined score in [0,1] intended for ranking edges; higher means
        stronger support for the edge.
    llm_score : float
        The input LLM score (returned unchanged for convenience).
    support_score : float
        A normalised support score derived from the retrieval scores using
        ``rag_mod.support_score_from_scores``.

    Notes
    -----
    The final score is a convex combination of a linear term and a
    multiplicative (geometric) term.  Hyperparameters controlling the
    combination live in ``modules.config``:
        - COMBINE_LAMBDA_LINEAR (λ) for the linear term
        - COMBINE_ALPHA_MULT  (α) for the multiplicative term
        - COMBINE_BETA        (β) to interpolate between linear and multiplicative
        - COMBINE_EPS         (ϵ) to avoid log(0) in the multiplicative term
    By default β=0, so only the linear term is used.
    """
    # Compute support_score from retrieval_scores
    if retrieval_scores:
        try:
            support_score = rag_mod.support_score_from_scores(retrieval_scores, method="max")
        except Exception:
            # Fall back to simple normalisation if support_score_from_scores fails
            max_abs = max(abs(s) for s in retrieval_scores) or 1.0
            support_score = max(retrieval_scores) / max_abs
    else:
        support_score = 0.0

    # Clip both scores into [0,1]
    support_score = max(0.0, min(1.0, float(support_score)))
    llm_score = max(0.0, min(1.0, float(llm_score))) if llm_score is not None else 0.0

    # Retrieve combination hyperparameters from the config
    from . import config as cfg
    lambda_ = getattr(cfg, "COMBINE_LAMBDA_LINEAR", 0.5)
    alpha_ = getattr(cfg, "COMBINE_ALPHA_MULT", 0.5)
    beta_ = getattr(cfg, "COMBINE_BETA", 0.0)
    eps_ = getattr(cfg, "COMBINE_EPS", 1e-8)

    # Linear term: λ*L + (1-λ)*S
    linear_term = lambda_ * llm_score + (1.0 - lambda_) * support_score

    # Multiplicative term: (L+ϵ)**α * (S+ϵ)**(1-α)
    mult_term = ((llm_score + eps_) ** alpha_) * ((support_score + eps_) ** (1.0 - alpha_))

    # Interpolate between linear and multiplicative
    final_score = (1.0 - beta_) * linear_term + beta_ * mult_term

    # Clip final score into [0,1]
    final_score = max(0.0, min(1.0, final_score))

    return final_score, llm_score, support_score


# -------------------------------------------------------------------
# Helper: build sentence-level evidence passages from retrieved docs
# -------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _build_passage_evidence_from_ret_docs(
    tf: str,
    tgt: str,
    ctx: str,
    ret_docs: List[Any],
    max_sent_per_doc: int = 3,
    max_docs: int = 5,
) -> str:
    """
    Simple fallback evidence builder when rag_mod.build_rag_context()
    returns an empty string.

    For each retrieved document we:
      * load the cached PubMed JSON,
      * concatenate title + abstract,
      * split into sentences,
      * first select up to max_sent_per_doc sentences that mention BOTH TF
        and target (case-insensitive),
      * if none are found, fall back to sentences that mention TF OR target,
      * prepend the PMID to each sentence.

    Returns
    -------
    A single HTML-ready string with one or more passage lines separated
    by <br>, or an empty string if no suitable passages could be
    constructed.
    """
    tf_l = tf.lower()
    tgt_l = tgt.lower()

    passages: List[str] = []

    for d in ret_docs[:max_docs]:
        pmid = getattr(d, "doc_id", None)
        if pmid is None:
            continue

        try:
            rec = _load_cached_pubmed_json(str(pmid))
            full_txt = _concat_title_abs(rec)
        except Exception:
            continue

        if not full_txt:
            continue

        sentences = _SENT_SPLIT_RE.split(full_txt)

        # Pass 1: sentences with BOTH TF and target
        selected: List[str] = []
        for sent in sentences:
            s_l = sent.lower()
            if tf_l in s_l and tgt_l in s_l:
                sent = sent.strip()
                if not sent:
                    continue
                selected.append(sent)
                if len(selected) >= max_sent_per_doc:
                    break

        # Pass 2: fallback to TF OR target if nothing selected
        if not selected:
            for sent in sentences:
                s_l = sent.lower()
                if tf_l in s_l or tgt_l in s_l:
                    sent = sent.strip()
                    if not sent:
                        continue
                    selected.append(sent)
                    if len(selected) >= max_sent_per_doc:
                        break

        for sent in selected:
            passages.append(f"[PMID {pmid}] {sent}")

    # Use <br> so each sentence appears on its own line in HTML
    return "<br>".join(passages).strip()

# -------------------------------------------------------------------
# NEW: Retrieval-centric RAGulate inference (production mode)
# -------------------------------------------------------------------

def run_ragulate_inference(
    edges_df: pd.DataFrame,
    method_type: str = "hybrid",
    encoder_name: Optional[str] = None,
    top_k_docs: int = config.TOP_K_RETRIEVE,
    max_sent_per_doc: int = config.MAX_SENT_PER_PAPER,
    use_llm: bool = False,
    llm_name: Optional[str] = None,
    classify: bool = False,
    use_aliases: bool = False,
    alias_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run RAGulate in 'production' mode on a set of predicted edges.

    This is the retrieval-centric pipeline:

      * For each (tf, target, context):
          - Build a natural-language query.
          - Retrieve PubMed documents via the chosen backend
            (bm25 / vector / hybrid / vanilla_rag).
          - Aggregate retrieval scores into a support_score.
          - Build a RAG context / evidence snippet.

      * Optionally, call an LLM (e.g. BioGPT) to summarise the
        evidence into a short textual rationale (llm_summary).

    Parameters
    ----------
    edges_df:
        DataFrame with columns at least: 'tf', 'target', 'context'.
    method_type:
        One of:
          - 'bm25'
          - 'vector'
          - 'hybrid'        (recommended: BM25 + MiniLM)
          - 'vanilla_rag'   (LlamaIndex or similar backend)
    encoder_name:
        Encoder name for 'vector' / 'hybrid' retrieval.
        None -> default MiniLM encoder inside retrieval.
    top_k_docs:
        Number of top documents to retrieve per edge for evidence.
    max_sent_per_doc:
        Maximum number of sentences per paper in the RAG context.
        (Actual behaviour depends on rag_mod.build_rag_context.)
    use_llm:
        If True, call an LLM to generate a brief rationale based on
        the evidence snippets.
    llm_name:
        Name of the LLM to use. If None and use_llm is True, defaults
        to config.BIOGPT_MODEL_NAME.
    classify:
        If True, in addition to retrieval and (optional) summarisation,
        the function will ask the LLM a strict yes/no question over the
        same evidence, combine the LLM probability with the retrieval
        support via ``_combine_llm_and_retrieval_scores``, and return
        extra columns:
          - ``score``: final combined confidence score.
          - ``llm_score``: raw yes/no probability from the LLM.
          - ``raw_answer``: raw text / token from the yes/no call.
    use_aliases:
        If True and method_type == 'hybrid', expand TF and target symbols
        using the alias expansion logic (BM25 + cosine + alias), matching
        your original RAGulate flagship model.
    alias_max:
        Maximum number of aliases per symbol when use_aliases=True.
        If None, falls back to config.MAX_ALIAS_PER_SYMBOL.

    Returns
    -------
    pd.DataFrame with columns:
        tf, target, context,
        support_score,
        n_support_papers,
        retrieved_pmids,
        retrieval_scores,
        evidence_snippets,
        llm_summary (optional; None if use_llm=False)

        If classify=True, the following additional columns are also
        included:
        score, llm_score, raw_answer.
    """
    # Prepare LLM if requested (for summary and/or classification)
    llm_tok, llm_mdl = None, None
    if use_llm or classify:
        if llm_name is None:
            llm_name = getattr(config, "BIOGPT_MODEL_NAME", None)
        if llm_name is None:
            raise ValueError(
                "LLM requested (use_llm=True or classify=True) but no llm_name "
                "and BIOGPT_MODEL_NAME not set in config"
            )

        if llm_name == config.BIOGPT_MODEL_NAME:
            llm_tok, llm_mdl = get_biogpt()
        elif llm_name == config.MISTRAL_MODEL_NAME:
            llm_tok, llm_mdl = get_mistral()
        elif llm_name == config.LLAMA31_MODEL_NAME:
            llm_tok, llm_mdl = get_llama31()
        elif llm_name == config.PHI3_MODEL_NAME:
            llm_tok, llm_mdl = get_phi3()
        elif llm_name == config.QWEN25_MODEL_NAME:
            llm_tok, llm_mdl = get_qwen25()
        else:
            raise ValueError(f"Unknown LLM name for inference: {llm_name}")

    rows_out: List[Dict[str, Any]] = []

    for _, r in edges_df.iterrows():
        tf = str(r["tf"])
        tgt = str(r["target"])
        ctx = str(r.get("context", ""))

        # 1) Build query from edge (possibly alias-expanded for hybrid)
        query = rag_mod.build_query_from_edge(tf, tgt, ctx)

        if method_type == "hybrid" and use_aliases:
            max_aliases = alias_max or getattr(config, "MAX_ALIAS_PER_SYMBOL", 4)

            def _alias_expr(sym: str) -> str:
                exps = alias_mod.expand_symbol_for_query(
                    sym,
                    include_self=True,
                    max_aliases=max_aliases,
                )
                exps = [s for s in exps if s]
                if not exps:
                    return sym
                if len(exps) == 1:
                    return exps[0]
                return "(" + " OR ".join(exps) + ")"

            tf_expr = _alias_expr(tf)
            tgt_expr = _alias_expr(tgt)
            query = f"{tf_expr} regulates {tgt_expr} in {ctx}"

        # 2) Retrieval
        if method_type == "bm25":
            ret_docs = retrieval.retrieve_bm25(query, top_k=top_k_docs)
        elif method_type == "vector":
            ret_docs = retrieval.retrieve_vector(
                query,
                encoder_name=encoder_name,
                top_k=top_k_docs,
            )
        elif method_type == "hybrid":
            ret_docs = retrieval.retrieve_hybrid_bm25_vector(
                query,
                encoder_name=encoder_name,  # None -> default MiniLM
                top_k=top_k_docs,
            )
        elif method_type == "vanilla_rag":
            ret_docs = retrieval.retrieve_vanilla_rag(query, top_k=top_k_docs)
        else:
            raise ValueError(
                f"Unsupported method_type for run_ragulate_inference: {method_type!r}"
            )

        retrieved_pmids: List[str] = []
        retrieval_scores: List[float] = []
        for d in ret_docs:
            pmid = getattr(d, "doc_id", None)
            score = float(getattr(d, "score", 0.0))
            if pmid is not None:
                retrieved_pmids.append(pmid)
                retrieval_scores.append(score)

        n_support_papers = len(retrieved_pmids)

        # 4) Evidence snippets / RAG context
        #    Try to use build_rag_context; if it returns empty, fall back to
        #    sentence-level passages extracted from title+abstracts; and only
        #    then to a fully generic text.
        try:
            # Prefer signature with max_sent_per_doc if available
            try:
                evidence_snippets = rag_mod.build_rag_context(
                    tf,
                    tgt,
                    ctx,
                    ret_docs,
                    max_sent_per_doc=max_sent_per_doc,
                )
            except TypeError:
                # Older signature without max_sent_per_doc
                evidence_snippets = rag_mod.build_rag_context(
                    tf,
                    tgt,
                    ctx,
                    ret_docs,
                )
        except Exception:
            evidence_snippets = ""

        # If still empty/blank, try our passage-based fallback
        if not isinstance(evidence_snippets, str) or not evidence_snippets.strip():
            evidence_snippets = _build_passage_evidence_from_ret_docs(
                tf=tf,
                tgt=tgt,
                ctx=ctx,
                ret_docs=ret_docs,
                max_sent_per_doc=max_sent_per_doc,
                max_docs=5,
            )

        # If that *also* fails, fall back to a minimal textual summary
        if not isinstance(evidence_snippets, str) or not evidence_snippets.strip():
            if retrieved_pmids:
                pmid_str = ", ".join(str(p) for p in retrieved_pmids[:10])
                evidence_snippets = (
                    f"Evidence for {tf} regulating {tgt} in the context '{ctx}'. "
                    f"Retrieved PubMed IDs: {pmid_str}."
                )
            else:
                evidence_snippets = (
                    f"Evidence for {tf} regulating {tgt} in the context '{ctx}'. "
                    "No supporting PubMed IDs were retrieved."
                )

        # 3 / 5) Support score + optional classification
        llm_score_val: Optional[float] = None
        final_score: Optional[float] = None
        raw_answer: Optional[str] = None

        if classify:
            if llm_tok is None or llm_mdl is None:
                raise ValueError(
                    "classify=True but LLM is not initialised. "
                    "Pass use_llm=True or a valid llm_name."
                )
            rag_prompt = rag_mod.build_rag_prompt(tf, tgt, ctx, evidence_snippets)
            llm_score_val, _, raw_answer = _answer_yes_no_single(
                llm_tok, llm_mdl, rag_prompt
            )
            final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(
                llm_score_val, retrieval_scores
            )
            support_score = float(support_score)
            llm_score_val = float(llm_prob)
        else:
            support_score = rag_mod.support_score_from_scores(
                retrieval_scores,
                method="max",
            )

        # 6) Optional LLM summary (free-text rationale, not classification)
        llm_summary = None
        if use_llm and llm_tok is not None and llm_mdl is not None:
            prompt = (
                "You are a biomedical domain expert.\n\n"
                f"Task: Based on the evidence below, summarise in 2–3 sentences whether "
                f"transcription factor {tf} regulates target gene {tgt} "
                f"in the context '{ctx}'. Indicate whether the evidence is direct, "
                "indirect, or ambiguous, and mention any key PMIDs if obvious.\n\n"
                "EVIDENCE:\n"
                f"{evidence_snippets}\n\n"
                "Answer in 2–3 sentences:"
            )

            # Allow a longer, configurable budget for summaries.
            # Falls back to 160 tokens if LLM_SUMMARY_MAX_NEW is not defined.
            summary_max_new = getattr(config, "LLM_SUMMARY_MAX_NEW", 160)

            full_text = _llm_generate(
                llm_tok,
                llm_mdl,
                prompt,
                max_new_tokens=summary_max_new,
            )

            # Some HF-style generators echo the prompt; strip it if present.
            if isinstance(full_text, str) and full_text.startswith(prompt):
                llm_summary = full_text[len(prompt):].strip()
            else:
                llm_summary = (full_text or "").strip()


        row: Dict[str, Any] = {
            "tf": tf,
            "target": tgt,
            "context": ctx,
            "support_score": float(support_score),
            "n_support_papers": int(n_support_papers),
            "retrieved_pmids": retrieved_pmids,
            "retrieval_scores": retrieval_scores,
            "evidence_snippets": evidence_snippets,
            "llm_summary": llm_summary,
        }

        if classify:
            row["llm_score"] = llm_score_val
            row["score"] = final_score
            row["raw_answer"] = raw_answer

        rows_out.append(row)

    return pd.DataFrame(rows_out)

def evaluate_pmid_hallucination(
    gold_df: pd.DataFrame,
    results_by_method: Dict[str, pd.DataFrame],
    pmid_col: str = "pmids",
) -> pd.DataFrame:
    """
    Compare different RAG/LLM methods on:
      - whether they mention the correct (gold) PMIDs
      - whether they hallucinate PMIDs that were never retrieved.

    Parameters
    ----------
    gold_df :
        Gold-standard edges with columns at least:
          'tf', 'target', 'context', and a PMID column (default: 'pmids').
        The PMID column may be a list or a string representation of a list.
    results_by_method :
        Dict mapping method_name -> DataFrame returned by run_ragulate_inference().
        Each results DF must contain columns:
          'tf', 'target', 'context', 'retrieved_pmids',
          and optionally 'llm_summary' and 'evidence_snippets'.
    pmid_col :
        Name of the gold_df column containing gold PMIDs.

    Returns
    -------
    pd.DataFrame with one row per method and columns:
        method
        n_edges
        n_with_gold_pmids
        n_with_summary_pmids
        n_with_gold_in_summary
        hallucination_rate
        strict_correct_rate
        mean_pmid_precision
        mean_pmid_recall
    """

    key_cols = ["tf", "target", "context"]

    # --- 1) Prepare gold PMIDs keyed by (tf, target, context) ---
    if pmid_col not in gold_df.columns:
        raise ValueError(f"gold_df must contain a '{pmid_col}' column with gold PMIDs")

    g = gold_df.copy()

    def _parse_pmids(val: Any) -> List[str]:
        pm_list = _maybe_eval_pmids_field(val)
        if pm_list is None:
            return []
        return [str(p) for p in pm_list if p]

    g["gold_pmids"] = g[pmid_col].apply(_parse_pmids)
    gold_key = (
        g[key_cols + ["gold_pmids"]]
        .drop_duplicates(subset=key_cols)
        .reset_index(drop=True)
    )

    # --- 2) Per-method evaluation ---
    summary_rows: List[Dict[str, Any]] = []

    for method_name, res_df in results_by_method.items():
        if res_df is None or res_df.empty:
            summary_rows.append(
                {
                    "method": method_name,
                    "n_edges": 0,
                    "n_with_gold_pmids": 0,
                    "n_with_summary_pmids": 0,
                    "n_with_gold_in_summary": 0,
                    "hallucination_rate": 0.0,
                    "strict_correct_rate": 0.0,
                    "mean_pmid_precision": 0.0,
                    "mean_pmid_recall": 0.0,
                }
            )
            continue

        df = res_df.copy()

        # Make sure we have the key cols
        missing_keys = [c for c in key_cols if c not in df.columns]
        if missing_keys:
            raise ValueError(
                f"results for method {method_name!r} are missing key columns: {missing_keys}"
            )

        # Merge in gold PMIDs
        merged = pd.merge(df, gold_key, on=key_cols, how="left")

        # Normalise retrieved_pmids to list[str]
        if "retrieved_pmids" not in merged.columns:
            merged["retrieved_pmids"] = [[] for _ in range(len(merged))]

        merged["retrieved_pmids_list"] = merged["retrieved_pmids"].apply(
            lambda x: [str(p) for p in (x or [])]
        )

        # Extract PMIDs from LLM summary if available
        if "llm_summary" in merged.columns:
            merged["summary_pmids"] = merged["llm_summary"].apply(_extract_pmids_from_text)
        else:
            merged["summary_pmids"] = [[] for _ in range(len(merged))]

        # Optionally: also extract from evidence_snippets
        if "evidence_snippets" in merged.columns:
            merged["evidence_pmids"] = merged["evidence_snippets"].apply(
                _extract_pmids_from_text
            )
        else:
            merged["evidence_pmids"] = [[] for _ in range(len(merged))]

        # --- Row-level stats ---
        n_edges = len(merged)
        n_with_gold_pmids = 0
        n_with_summary_pmids = 0
        n_with_gold_in_summary = 0
        n_hallucinated = 0
        n_strict_correct = 0

        precisions: List[float] = []
        recalls: List[float] = []

        for _, row in merged.iterrows():
            gold_set = set(row.get("gold_pmids") or [])
            ret_set = set(row.get("retrieved_pmids_list") or [])
            summ_set = set(row.get("summary_pmids") or [])

            has_gold = len(gold_set) > 0
            has_summary = len(summ_set) > 0

            if has_gold:
                n_with_gold_pmids += 1
            if has_summary:
                n_with_summary_pmids += 1

            if has_gold and has_summary and (gold_set & summ_set):
                n_with_gold_in_summary += 1

            # Hallucination = summary mentions any PMID not retrieved at all
            hallucinated = bool(summ_set - ret_set)
            if hallucinated:
                n_hallucinated += 1

            # Strict correctness: summary PMIDs non-empty and subset of gold PMIDs
            if has_summary and summ_set.issubset(gold_set):
                n_strict_correct += 1

            # Precision / recall over PMIDs (only when defined)
            if has_summary:
                inter = gold_set & summ_set
                denom_p = len(summ_set)
                if denom_p > 0:
                    precisions.append(len(inter) / denom_p)
            if has_gold:
                inter = gold_set & summ_set
                denom_r = len(gold_set)
                if denom_r > 0:
                    recalls.append(len(inter) / denom_r)

        hallucination_rate = float(n_hallucinated) / float(n_edges) if n_edges > 0 else 0.0
        strict_correct_rate = float(n_strict_correct) / float(n_edges) if n_edges > 0 else 0.0
        mean_p = float(np.mean(precisions)) if precisions else 0.0
        mean_r = float(np.mean(recalls)) if recalls else 0.0

        summary_rows.append(
            {
                "method": method_name,
                "n_edges": n_edges,
                "n_with_gold_pmids": n_with_gold_pmids,
                "n_with_summary_pmids": n_with_summary_pmids,
                "n_with_gold_in_summary": n_with_gold_in_summary,
                "hallucination_rate": hallucination_rate,
                "strict_correct_rate": strict_correct_rate,
                "mean_pmid_precision": mean_p,
                "mean_pmid_recall": mean_r,
            }
        )

    return pd.DataFrame(summary_rows)
# ---------------------------------------------------------------------------
# 8. PMCID / PMID hallucination comparison helpers
# ---------------------------------------------------------------------------
def _add_batched_llm_summaries(
    df: pd.DataFrame,
    llm_tok,
    llm_mdl,
    *,
    summary_max_new: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add an 'llm_summary' column to df in a batched way.

    Expects df to have columns: 'tf', 'target', 'context', 'evidence_snippets'.
    Uses the same summary prompt style as run_ragulate_inference, but
    runs all prompts through _llm_generate_batch instead of a per-row loop.
    """
    if df is None or df.empty:
        df = df.copy()
        df["llm_summary"] = None
        return df

    if llm_tok is None or llm_mdl is None:
        # No LLM available -> just fill with empty summaries
        df = df.copy()
        df["llm_summary"] = None
        return df

    if summary_max_new is None:
        summary_max_new = getattr(config, "LLM_SUMMARY_MAX_NEW", 160)

    prompts: List[str] = []
    for _, row in df.iterrows():
        tf = str(row.get("tf", ""))
        tgt = str(row.get("target", ""))
        ctx = str(row.get("context", ""))
        evidence = row.get("evidence_snippets", "")

        prompt = (
            "You are a biomedical domain expert.\n\n"
            f"Task: Based on the evidence below, summarise in 2–3 sentences whether "
            f"transcription factor {tf} regulates target gene {tgt} "
            f"in the context '{ctx}'. Indicate whether the evidence is direct, "
            "indirect, or ambiguous, and mention any key PMIDs if obvious.\n\n"
            "EVIDENCE:\n"
            f"{evidence}\n\n"
            "Answer in 2–3 sentences:"
        )
        prompts.append(prompt)

    # Batched generation
    full_texts = _llm_generate_batch(
        llm_tok,
        llm_mdl,
        prompts,
        max_new_tokens=summary_max_new,
        mode="raw",
    )

    cleaned: List[str] = []
    for prompt, full_text in zip(prompts, full_texts):
        if isinstance(full_text, str) and full_text.startswith(prompt):
            cleaned.append(full_text[len(prompt):].strip())
        else:
            cleaned.append((full_text or "").strip())

    out = df.copy()
    out["llm_summary"] = cleaned
    return out


def _run_llm_only_summaries(
    edges_df: pd.DataFrame,
    llm_tok,
    llm_mdl,
    *,
    summary_max_new: Optional[int] = None,
) -> pd.DataFrame:
    """
    LLM-only baseline for PMID hallucination evaluation.

    - No retrieval is performed.
    - retrieved_pmids is always [].
    - We just ask the LLM, based on its general knowledge, to comment on
      TF -> target regulation in context and mention PMIDs if it knows any.
    """
    if edges_df is None or edges_df.empty:
        return pd.DataFrame(
            columns=[
                "tf",
                "target",
                "context",
                "support_score",
                "n_support_papers",
                "retrieved_pmids",
                "retrieval_scores",
                "evidence_snippets",
                "llm_summary",
            ]
        )

    df = (
        edges_df[["tf", "target", "context"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    if summary_max_new is None:
        summary_max_new = getattr(config, "LLM_SUMMARY_MAX_NEW", 160)

    if llm_tok is None or llm_mdl is None:
        # No LLM available -> fill with blanks
        df["support_score"] = 0.0
        df["n_support_papers"] = 0
        df["retrieved_pmids"] = [[] for _ in range(len(df))]
        df["retrieval_scores"] = [[] for _ in range(len(df))]
        df["evidence_snippets"] = ""
        df["llm_summary"] = None
        return df

    prompts: List[str] = []
    for _, row in df.iterrows():
        tf = str(row.get("tf", ""))
        tgt = str(row.get("target", ""))
        ctx = str(row.get("context", ""))

        prompt = (
            "You are a biomedical domain expert.\n\n"
            f"Task: Based on your general scientific knowledge (without any provided documents), "
            f"summarise in 2–3 sentences whether transcription factor {tf} regulates "
            f"target gene {tgt} in the context '{ctx}'. "
            "If you mention PubMed references, use the pattern 'PMID 12345678'. "
            "If you are not sure, say that explicitly.\n\n"
            "Answer in 2–3 sentences:"
        )
        prompts.append(prompt)

    full_texts = _llm_generate_batch(
        llm_tok,
        llm_mdl,
        prompts,
        max_new_tokens=summary_max_new,
        mode="raw",
    )

    summaries: List[str] = []
    for prompt, full_text in zip(prompts, full_texts):
        if isinstance(full_text, str) and full_text.startswith(prompt):
            summaries.append(full_text[len(prompt):].strip())
        else:
            summaries.append((full_text or "").strip())

    df["support_score"] = 0.0
    df["n_support_papers"] = 0
    df["retrieved_pmids"] = [[] for _ in range(len(df))]
    df["retrieval_scores"] = [[] for _ in range(len(df))]
    df["evidence_snippets"] = ""
    df["llm_summary"] = summaries

    return df

def run_direct_llm_inference_for_hallucination(
    edges_df: pd.DataFrame,
    *,
    llm_name: Optional[str] = None,
    summary_max_new_tokens: int = 320,
) -> pd.DataFrame:
    """LLM-only baseline for PMID hallucination experiments.

    This **does not** perform any retrieval. For each (tf, target, context)
    triple it:
      * asks the LLM a short yes/no-style question
      * requests a 2–3 sentence justification where the model may optionally
        mention PubMed IDs
      * returns a DataFrame with the same key columns used by
        :func:`evaluate_pmid_hallucination`, but with an empty
        ``retrieved_pmids`` list (so every PMID mentioned is counted
        as hallucinated).
    """
    if llm_name is None:
        llm_name = config.MISTRAL_MODEL_NAME

    llm_tok, llm_mdl = get_llm_and_tokenizer(llm_name)

    records: List[Dict[str, Any]] = []

    # We only need the key columns; keep other cols if present, but don't rely on them.
    for _, row in edges_df.iterrows():
        tf = str(row.get("tf", "")).strip()
        target = str(row.get("target", "")).strip()
        context = str(row.get("context", "")).strip()

        if not tf or not target:
            continue

        prompt = rag_mod.build_direct_question_prompt(tf, target, context)
        # Encourage (but do not force) the model to mention PMIDs when it
        # genuinely recalls them. Any such PMIDs will be treated as
        # hallucinated because there is no retrieval in this baseline.
        prompt = (
            prompt
            + "\n\n"
            + "In 2–3 sentences, briefly justify your answer. "
              "If you recall specific PubMed IDs that directly support your "
              "answer, mention them explicitly in the form 'PMID 12345678'. "
              "If you are not sure about any PubMed IDs, say so and do not "
              "invent them."
        )

        answer = _llm_generate(
            llm_tok,
            llm_mdl,
            prompt,
            max_new_tokens=summary_max_new_tokens,
        )

        records.append(
            {
                "tf": tf,
                "target": target,
                "context": context,
                # No retrieval in this baseline
                "retrieved_pmids": [],
                "n_support_papers": 0,
                # We treat the generated justification as the "summary"
                "llm_summary": answer,
                # And mirror it into evidence_snippets so downstream code can
                # still inspect it if desired
                "evidence_snippets": answer,
            }
        )

    return pd.DataFrame.from_records(records)



def run_pmid_hallucination_suite(
    gold_df: pd.DataFrame,
    *,
    pmid_col: str = "pmids",
    llm_name: Optional[str] = None,
    top_k_docs: int = config.TOP_K_RETRIEVE,
) -> Dict[str, Any]:
    """
    Compare three configurations on PMID faithfulness:

      1) 'ragulate_hybrid_alias' : flagship hybrid BM25 + MiniLM + HGNC aliases
      2) 'vanilla_rag'           : naive / LlamaIndex-style RAG backend
      3) 'llm_only'              : direct LLM (no retrieval), to expose hallucinations

    All LLM summaries are generated in batched mode to keep this fast
    even for hundreds/thousands of edges.

    Returns
    -------
    {
      "summary": pd.DataFrame (from evaluate_pmid_hallucination),
      "results_by_method": {
          "ragulate_hybrid_alias": df1,
          "vanilla_rag": df2,
          "llm_only": df3,
      }
    }
    """
    # Make sure gold has the PMID column
    if pmid_col not in gold_df.columns:
        raise ValueError(
            f"gold_df must contain a '{pmid_col}' column for PMID hallucination evaluation"
        )

    # Unique edges used for inference
    edges_df = (
        gold_df[["tf", "target", "context"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Initialise the summarisation LLM once
    # ----------------------------------------------------------
    if llm_name is None:
        llm_name = getattr(config, "MISTRAL_MODEL_NAME", None)

    if llm_name is None:
        raise ValueError(
            "No llm_name provided and MISTRAL_MODEL_NAME not set in config"
        )

    if llm_name == config.BIOGPT_MODEL_NAME:
        llm_tok, llm_mdl = get_biogpt()
    elif llm_name == config.MISTRAL_MODEL_NAME:
        llm_tok, llm_mdl = get_mistral()
    elif llm_name == config.LLAMA31_MODEL_NAME:
        llm_tok, llm_mdl = get_llama31()
    elif llm_name == config.PHI3_MODEL_NAME:
        llm_tok, llm_mdl = get_phi3()
    elif llm_name == config.QWEN25_MODEL_NAME:
        llm_tok, llm_mdl = get_qwen25()
    else:
        raise ValueError(f"Unknown LLM name for hallucination suite: {llm_name}")

    summary_max_new = getattr(config, "LLM_SUMMARY_MAX_NEW", 160)

    # ----------------------------------------------------------
    # 1) RAGulate flagship: hybrid + aliases, retrieval-only
    # ----------------------------------------------------------
    print("RAGulate flagship: hybrid + aliases + LLM summaries")
    ragulate_flag_df = run_ragulate_inference(
        edges_df=edges_df,
        method_type="hybrid",
        encoder_name=None,  # None -> default MiniLM in retrieve_hybrid_bm25_vector
        top_k_docs=top_k_docs,
        max_sent_per_doc=config.MAX_SENT_PER_PAPER,
        use_llm=False,#True,                  
        llm_name=None, #config.MISTRAL_MODEL_NAME,
        classify=False,
        use_aliases=True,
        alias_max=getattr(config, "MAX_ALIAS_PER_SYMBOL", 4),
    )

    ragulate_flag_df = _add_batched_llm_summaries(
        ragulate_flag_df,
        llm_tok,
        llm_mdl,
        summary_max_new=summary_max_new,
    )

    # ----------------------------------------------------------
    # 2) Vanilla RAG: LlamaIndex or other backend
    # ----------------------------------------------------------
    print("Naive RAG / vanilla RAG (LlamaIndex-style retriever)")
    vanilla_rag_df = run_ragulate_inference(
        edges_df=edges_df,
        method_type="vanilla_rag",
        encoder_name="default",
        top_k_docs=top_k_docs,
        max_sent_per_doc=config.MAX_SENT_PER_PAPER,
        use_llm=False,#True,                  
        llm_name=None,#config.MISTRAL_MODEL_NAME,
        classify=False,
        use_aliases=False,
        alias_max=None,
    )

    vanilla_rag_df = _add_batched_llm_summaries(
        vanilla_rag_df,
        llm_tok,
        llm_mdl,
        summary_max_new=summary_max_new,
    )

    # ----------------------------------------------------------
    # 3) LLM-only baseline (no retrieval)
    # ----------------------------------------------------------
    print("LLM-only baseline (no retrieval; just hallucinated PMIDs)")
    llm_only_df = _run_llm_only_summaries(
        edges_df,
        llm_tok,
        llm_mdl,
        summary_max_new=summary_max_new,
    )

    # ----------------------------------------------------------
    # Package results and compute evaluation
    # ----------------------------------------------------------
    results_by_method: Dict[str, pd.DataFrame] = {
        "ragulate_hybrid_alias": ragulate_flag_df,
        "vanilla_rag": vanilla_rag_df,
        "llm_only": llm_only_df,
    }

    summary_df = evaluate_pmid_hallucination(
        gold_df=gold_df,
        results_by_method=results_by_method,
        pmid_col=pmid_col,
    )

    return {
        "summary": summary_df,
        "results_by_method": results_by_method,
    }




def run_llm_only_inference(
    edges_df: pd.DataFrame,
    llm_name: Optional[str] = None,
    classify: bool = False,
) -> pd.DataFrame:
    """
    LLM-only baseline: no retrieval, no explicit evidence.
    We just ask the LLM directly about each TF–target–context edge and
    optionally do yes/no classification.

    Returns a DataFrame with the same core shape as run_ragulate_inference,
    so it can be passed into evaluate_pmid_hallucination().
    """
    # Init LLM
    if llm_name is None:
        llm_name = getattr(config, "MISTRAL_MODEL_NAME", None)
    if llm_name is None:
        raise ValueError("llm_name must be provided or MISTRAL_MODEL_NAME set in config")

    if llm_name == config.BIOGPT_MODEL_NAME:
        llm_tok, llm_mdl = get_biogpt()
    elif llm_name == config.MISTRAL_MODEL_NAME:
        llm_tok, llm_mdl = get_mistral()
    elif llm_name == config.LLAMA31_MODEL_NAME:
        llm_tok, llm_mdl = get_llama31()
    elif llm_name == config.PHI3_MODEL_NAME:
        llm_tok, llm_mdl = get_phi3()
    elif llm_name == config.QWEN25_MODEL_NAME:
        llm_tok, llm_mdl = get_qwen25()
    else:
        raise ValueError(f"Unknown LLM name for LLM-only inference: {llm_name}")

    rows_out: List[Dict[str, Any]] = []

    for _, r in edges_df.iterrows():
        tf = str(r["tf"])
        tgt = str(r["target"])
        ctx = str(r.get("context", ""))

        # -----------------------------
        # 1) Optional yes/no classification (no retrieval scores)
        # -----------------------------
        llm_score_val: Optional[float] = None
        final_score: Optional[float] = None
        raw_answer: Optional[str] = None
        support_score: float = 0.0  # no retrieval

        if classify:
            q_prompt = rag_mod.build_direct_question_prompt(tf, tgt, ctx)
            llm_score_val, _, raw_answer = _answer_yes_no_single(
                llm_tok, llm_mdl, q_prompt
            )
            final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(
                llm_score_val, []
            )
            support_score = float(support_score)
            llm_score_val = float(llm_prob)

        # -----------------------------
        # 2) Free-text summary (where PMIDs may be hallucinated)
        # -----------------------------
        summary_prompt = (
            "You are a biomedical domain expert.\n\n"
            f"Question: Does transcription factor {tf} regulate target gene {tgt} "
            f"in the context '{ctx}'?\n"
            "If you know of PubMed evidence, mention key PMIDs explicitly in the form "
            "'PMID 12345678'. If you are not sure, say so.\n\n"
            "Answer in 2–3 sentences:"
        )

        summary_max_new = getattr(config, "LLM_SUMMARY_MAX_NEW", 160)
        full_text = _llm_generate(
            llm_tok,
            llm_mdl,
            summary_prompt,
            max_new_tokens=summary_max_new,
        )
        if isinstance(full_text, str) and full_text.startswith(summary_prompt):
            llm_summary = full_text[len(summary_prompt):].strip()
        else:
            llm_summary = (full_text or "").strip()

        row: Dict[str, Any] = {
            "tf": tf,
            "target": tgt,
            "context": ctx,
            "support_score": float(support_score),
            "n_support_papers": 0,
            "retrieved_pmids": [],       # <- crucial for hallucination metric
            "retrieval_scores": [],
            "evidence_snippets": "(no retrieval; LLM answered directly).",
            "llm_summary": llm_summary,
        }

        if classify:
            row["llm_score"] = llm_score_val
            row["score"] = final_score
            row["raw_answer"] = raw_answer

        rows_out.append(row)

    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------
# Export helpers: CSV + HTML with PubMed links and highlighted evidence
# ---------------------------------------------------------------------

def save_ragulate_results_csv(df: pd.DataFrame, csv_path: str) -> None:
    """
    Save the full RAGulate inference results table to CSV.

    Parameters
    ----------
    df:
        DataFrame returned by run_ragulate_inference.
    csv_path:
        Output path for the CSV file.
    """
    df.to_csv(csv_path, index=False)


def _link_pmids_for_html(pmids) -> str:
    """
    Turn a list of PMIDs into HTML hyperlinks for the HTML export.

    Example:
        ['41308124', '12345'] ->
        '<a href="https://pubmed.ncbi.nlm.nih.gov/41308124/">41308124</a> ...'

    Note: currently not used in save_ragulate_results_html, but kept
    for potential future use.
    """
    if not isinstance(pmids, (list, tuple)):
        return ""

    links = []
    for pmid in pmids:
        if pmid is None or pmid == "":
            continue
        pmid_str = str(pmid).strip()
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"
        links.append(f'<a href="{url}" target="_blank">{pmid_str}</a>')
    return " ".join(links)


def _highlight_terms_in_text(text, terms) -> str:
    """
    Highlight TF, target, context and key regulation verbs within the
    evidence text using <mark>...</mark>. Case-insensitive,
    word-boundary matches.
    """
    # Robust to NaN / None / non-string
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    # Clean and deduplicate terms
    clean_terms = {t for t in terms if isinstance(t, str) and t.strip()}

    # Also highlight some common regulation verbs
    verbs = [
        "regulates",
        "induces",
        "activates",
        "represses",
        "upregulates",
        "downregulates",
    ]
    clean_terms.update(verbs)

    for term in clean_terms:
        # Match the term case-insensitively with word boundaries
        pattern = re.compile(rf"\b({re.escape(term)})\b", flags=re.IGNORECASE)

        # Use a lambda so we can inject the matched text directly
        text = pattern.sub(lambda m: f"<mark>{m.group(1)}</mark>", text)

    return text


PMID_RE = re.compile(r"\bPMID\s+(\d+)\b", flags=re.IGNORECASE)

def _extract_pmids_from_text(text: Any) -> List[str]:
    """
    Extract PMIDs mentioned as 'PMID 12345678' (case-insensitive) from
    arbitrary text. Returns a list of strings.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    if not isinstance(text, str):
        text = str(text)
    return [m.group(1) for m in PMID_RE.finditer(text)]


def _link_pmids_in_text(text: str) -> str:
    """
    Turn '[PMID 12345678]' patterns inside a text blob into clickable
    PubMed hyperlinks while preserving the surrounding brackets.

    Example:
        "[PMID 12345678] some sentence"
        ->
        "[<a href='https://pubmed.ncbi.nlm.nih.gov/12345678/' target='_blank'>PMID 12345678</a>] some sentence"
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    pattern = re.compile(r"\[PMID\s+(\d+)\]")

    def _repl(match: re.Match) -> str:
        pmid = match.group(1)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return f'[<a href="{url}" target="_blank">PMID {pmid}</a>]'

    return pattern.sub(_repl, text)


def _wrap_evidence_with_toggle(html_text: str, max_blocks: int = 3) -> str:
    """
    Wrap the (already-highlighted) evidence HTML into a short/long
    toggle block.

    We treat '<br>'-separated segments as "sentences" or atomic
    evidence units. The first `max_blocks` are shown by default,
    with a compact note indicating how many additional sentences
    are hidden. A "Show more / Show less" link toggles the view.
    """
    if not isinstance(html_text, str):
        html_text = "" if html_text is None else str(html_text)

    # Split on <br> (any of <br>, <br/>, <br />).
    blocks = re.split(r"<br\s*/?>", html_text)
    # Keep non-empty blocks
    blocks = [b for b in blocks if b.strip()]

    # If there isn't enough content to warrant a toggle, return as-is.
    if max_blocks <= 0 or len(blocks) <= max_blocks:
        return html_text

    short_blocks = blocks[:max_blocks]
    full_blocks = blocks

    short_html = "<br>".join(short_blocks)
    full_html = "<br>".join(full_blocks)

    n_more = len(full_blocks) - max_blocks
    more_label = "sentence" if n_more == 1 else "sentences"

    # Wrap both versions in a single HTML snippet with a toggle link.
    wrapped = (
        '<div class="evidence-wrapper">'
        f'<div class="evidence-short">{short_html} '
        f'<span class="evidence-more-note">… ({n_more} more {more_label})</span>'
        "</div>"
        f'<div class="evidence-full" style="display:none;">{full_html}</div>'
        '<a href="#" class="toggle-evidence" data-state="short">Show more</a>'
        "</div>"
    )
    return wrapped


def save_ragulate_results_html(
    df: pd.DataFrame,
    html_path: str,
    title: str = "RAGulate Results",
) -> None:
    """
    Export RAGulate results to an HTML table with:
      - PubMed IDs mentioned as [PMID ...] in the evidence column
        turned into clickable hyperlinks.
      - TF / target / context and key regulation verbs highlighted
        in the evidence column.
      - LLM summary shown as a separate column (if present).
      - Evidence column shortened per row with a small "show more"
        toggle to reveal the full text.

    Parameters
    ----------
    df:
        DataFrame returned by run_ragulate_inference.
    html_path:
        Output path for the HTML file.
    title:
        Title shown at the top of the HTML page.
    """
    html_df = df.copy().reset_index(drop=True)

    # --- Normalize newlines in llm_summary so the browser shows line breaks ---
    if "llm_summary" in html_df.columns:
        col = html_df["llm_summary"].fillna("").astype(str)
        # Handle both real newlines and literal "\n" sequences
        col = col.str.replace("\r\n", "\n", regex=False)
        col = col.str.replace("\\n\\n", "<br><br>", regex=False)  # literal "\n\n"
        col = col.str.replace("\\n", "<br>", regex=False)         # literal "\n"
        col = col.str.replace("\n\n", "<br><br>", regex=False)    # actual blank line
        col = col.str.replace("\n", "<br>", regex=False)          # actual newline
        html_df.loc[:, "llm_summary"] = col
        
    # Ensure evidence_snippets exists
    if "evidence_snippets" not in html_df.columns:
        html_df["evidence_snippets"] = ""

    # Add column with highlighted evidence based directly on evidence_snippets
    def _make_evidence(row) -> str:
        base_text = row["evidence_snippets"]
        highlighted = _highlight_terms_in_text(
            base_text,
            [row.get("tf", ""), row.get("target", ""), row.get("context", "")],
        )
        # Make [PMID ...] clickable inside the highlighted text
        highlighted = _link_pmids_in_text(highlighted)
        # Wrap with a short/long toggle (first 3 blocks by default)
        highlighted = _wrap_evidence_with_toggle(highlighted, max_blocks=3)
        return highlighted

    html_df["evidence_highlighted"] = html_df.apply(_make_evidence, axis=1)

    # Decide which score column to show
    score_col = "score" if "score" in html_df.columns else "support_score"

    cols_for_html = [
        "tf",
        "target",
        "context",
        score_col,
    ]

    # Do NOT show pred_label in the HTML export (even if present)
    # No pmid_links column either; PMIDs are clickable directly in evidence_highlighted.

    cols_for_html.append("evidence_highlighted")

    if "llm_summary" in html_df.columns:
        cols_for_html.append("llm_summary")

    # Keep only columns that actually exist
    cols_for_html = [c for c in cols_for_html if c in html_df.columns]

    html_table_df = html_df[cols_for_html]

    table_html = html_table_df.to_html(
        index=False,
        escape=False,  # allow <a> and <mark> tags
        border=1,
        justify="left",
    )

    html_page = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
      body {{
        font-family: Arial, sans-serif;
        margin: 20px;
      }}
      table {{
        border-collapse: collapse;
        width: 100%;
      }}
      th, td {{
        padding: 6px 8px;
        border: 1px solid #ccc;
        vertical-align: top;
        word-wrap: break-word;
      }}
      mark {{
        background-color: #ffec99;
        font-weight: bold;
      }}
      /* Make the score column (4th) narrower so more width goes to text columns */
      th:nth-child(4), td:nth-child(4) {{
        white-space: nowrap;
        width: 80px;
      }}
      /* Give the last column (typically llm_summary) extra width */
      th:last-child, td:last-child {{
        width: 35%;
      }}
      .evidence-wrapper {{
        position: relative;
      }}
      .evidence-short, .evidence-full {{
        white-space: normal;
      }}
      .evidence-more-note {{
        font-size: 0.9em;
        color: #555;
      }}
      .toggle-evidence {{
        display: inline-block;
        margin-top: 4px;
        font-size: 0.9em;
        cursor: pointer;
        color: #0066cc;
        text-decoration: none;
      }}
      .toggle-evidence:hover {{
        text-decoration: underline;
      }}
    </style>
    </head>
    <body>
    <h2>{title}</h2>
    {table_html}
    <script>
      // Simple show more / show less toggle for evidence cells
      document.addEventListener('click', function (event) {{
        if (event.target && event.target.classList.contains('toggle-evidence')) {{
          event.preventDefault();
          var link = event.target;
          var wrapper = link.closest('.evidence-wrapper');
          if (!wrapper) return;
          var shortDiv = wrapper.querySelector('.evidence-short');
          var fullDiv = wrapper.querySelector('.evidence-full');
          if (!shortDiv || !fullDiv) return;
          var state = link.getAttribute('data-state') || 'short';
          if (state === 'short') {{
            shortDiv.style.display = 'none';
            fullDiv.style.display = 'block';
            link.textContent = 'Show less';
            link.setAttribute('data-state', 'full');
          }} else {{
            shortDiv.style.display = 'block';
            fullDiv.style.display = 'none';
            link.textContent = 'Show more';
            link.setAttribute('data-state', 'short');
          }}
        }}
      }});
    </script>
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_page)

# ---------- Retrieval-only experiment (for benchmarks) ----------


def run_retrieval_experiment(
    gold_df: pd.DataFrame,
    method_type: str,
    encoder_name: Optional[str],
    top_k: int,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
    positives_only: bool = False,
    query_mode: str = "tf_tg_ctx",      # "tf_tg_ctx" or "tf_tg_only"
    expand_context: bool = False,
    max_ctx_terms: int = 4,
    use_aliases: bool = False,          # <-- NEW
    alias_max: Optional[int] = None,    # <-- NEW (defaults to config.MAX_ALIAS_PER_SYMBOL)
) -> List[Dict[str, Any]]:
    """
    Run retrieval-only evaluation, returning rows like:
      {
        'tf': ...,
        'gene': ...,
        'context': ...,
        'gold_pmids': [...],
        'retrieved_pmids': [...],
        'retrieval_scores': [...],
        'support_score': float,
        'label': 0/1,
      }

    Parameters
    ----------
    positives_only :
        When True, restrict evaluation to edges with label == 1.
    query_mode :
        "tf_tg_ctx"  -> "<TF_expr> regulates <GENE_expr> in <context_phrase>"
        "tf_tg_only" -> "<TF_expr> regulates <GENE_expr>"
    expand_context :
        When True (and query_mode == "tf_tg_ctx"), use the existing
        KW-based context expansion logic to enrich the context phrase.
    use_aliases :
        When True, expand TF and target using HGNC aliases and OR them
        in the query, e.g. "(TP53 OR P53 OR LFS1) regulates (BAX OR ...)".
    alias_max :
        Max number of alias terms per symbol (excluding the original).
        If None, falls back to config.MAX_ALIAS_PER_SYMBOL.
    """
    if alias_max is None:
        alias_max = getattr(config, "MAX_ALIAS_PER_SYMBOL", 4)

    rows: List[Dict[str, Any]] = []

    def _symbol_expr(sym: str) -> str:
        """Return symbol or (sym1 OR sym2 ...) if use_aliases is enabled."""
        if not sym:
            return sym
        if not use_aliases:
            return sym

        expanded = alias_mod.expand_symbol_for_query(
            sym,
            include_self=True,
            max_aliases=alias_max,
        )
        expanded = [s for s in expanded if s]
        if not expanded:
            return sym
        if len(expanded) == 1:
            return expanded[0]
        return "(" + " OR ".join(expanded) + ")"

    for _, row in gold_df.iterrows():
        tf = row["tf"]
        tgt = row["target"]
        ctx = row["context"]

        lbl = int(row["label"]) if "label" in row.index else None
        if positives_only and lbl != 1:
            continue

        pmid_val = None
        if "pmids" in row.index:
            pmid_val = row["pmids"]
        elif "pubmed_ids" in row.index:
            pmid_val = row["pubmed_ids"]

        gold_ids = _maybe_eval_pmids_field(pmid_val) or []

        # Build TF / gene expressions (with or without aliases)
        tf_expr = _symbol_expr(tf)
        gene_expr = _symbol_expr(tgt)

        # Build the query according to query_mode / context options
        if query_mode == "tf_tg_only":
            query = f"{tf_expr} regulates {gene_expr}"
        elif query_mode == "tf_tg_ctx":
            # re-use your existing context phrase / expansion logic if you have it
            if expand_context and ctx:
                ctx_phrase = rag_mod.expand_context_terms(  # type: ignore[attr-defined]
                    ctx,
                    max_terms=max_ctx_terms,
                )
            else:
                ctx_phrase = ctx
            query = f"{tf_expr} regulates {gene_expr} in {ctx_phrase}"
        else:
            # Fallback: original RAGulate query builder (no alias ORs)
            query = rag_mod.build_query_from_edge(tf, tgt, ctx)

        # -----------------------------
        # Actual retrieval
        # -----------------------------
        if method_type == "bm25":
            ret_docs = retrieval.retrieve_bm25(query, top_k=top_k)
        elif method_type == "ragulate":
            ret_docs = retrieval.retrieve_vector(
                query,
                encoder_name=encoder_name,
                top_k=top_k,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
            )
        elif method_type == "vanilla_rag":
            ret_docs = retrieval.retrieve_vanilla_rag(query, top_k=top_k)
        elif method_type == "hybrid":
            ret_docs = retrieval.retrieve_hybrid_bm25_vector(
                query,
                encoder_name=encoder_name,
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unsupported method_type for retrieval: {method_type}")

        # Collect PMIDs + scores
        retrieved_pmids: List[str] = []
        retrieval_scores: List[float] = []
        for d in ret_docs:
            pmid = getattr(d, "doc_id", None)
            score = float(getattr(d, "score", 0.0))
            if pmid is not None:
                retrieved_pmids.append(pmid)
                retrieval_scores.append(score)

        support = rag_mod.support_score_from_scores(retrieval_scores, method="max")

        out_row: Dict[str, Any] = {
            "tf": tf,
            "gene": tgt,
            "context": ctx,
            "gold_pmids": gold_ids,
            "retrieved_pmids": retrieved_pmids,
            "retrieval_scores": retrieval_scores,
            "support_score": support,
        }
        if lbl is not None:
            out_row["label"] = lbl

        rows.append(out_row)

    return rows



def run_classification_experiment(
    gold_df: pd.DataFrame,
    method_type: str,
    encoder_name: Optional[str],
    llm_name: Optional[str],
    top_k: int,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
    use_aliases: bool = False,
    alias_max: Optional[int] = None,
) -> List[Dict[str, Any]]:


    rows: List[Dict[str, Any]] = []

    # Get tokenizer/model pair if needed
    llm_tok, llm_mdl = None, None
    if llm_name is not None:
        if llm_name == config.BIOGPT_MODEL_NAME:
            llm_tok, llm_mdl = get_biogpt()
        elif llm_name == config.MISTRAL_MODEL_NAME:
            llm_tok, llm_mdl = get_mistral()
        elif llm_name == config.LLAMA31_MODEL_NAME:
            llm_tok, llm_mdl = get_llama31()
        elif llm_name == config.PHI3_MODEL_NAME:
            llm_tok, llm_mdl = get_phi3()
        elif llm_name == config.QWEN25_MODEL_NAME:
            llm_tok, llm_mdl = get_qwen25()
        else:
            raise ValueError(f"Unknown LLM name: {llm_name}")

    for _, r in gold_df.iterrows():
        tf = r["tf"]
        tgt = r["target"]
        ctx = r["context"]
        label = int(r["label"])

        # NEW: pull pmids from the gold dataframe
        pmid_val = r.get("pmids", None)
        gold_pmids = _maybe_eval_pmids_field(pmid_val) or []

        if method_type == "direct_llm":
            out = run_direct_llm_edge(tf, tgt, ctx, llm_tok, llm_mdl)

        elif method_type == "ragulate":
            out = run_ragulate_edge(
                tf,
                tgt,
                ctx,
                tok=llm_tok,
                mdl=llm_mdl,
                encoder_name=encoder_name,
                top_k=top_k,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
            )

        elif method_type == "hybrid":
            out = run_hybrid_edge(
                tf,
                tgt,
                ctx,
                tok=llm_tok,
                mdl=llm_mdl,
                encoder_name=encoder_name,
                top_k=top_k,
                use_aliases=use_aliases,
                alias_max=alias_max,
            )

        elif method_type == "bm25_llm":
            out = run_bm25_llm_edge(tf, tgt, ctx, tok=llm_tok, mdl=llm_mdl, top_k=top_k)

        elif method_type == "bm25_simple":
            out = run_bm25_simple_edge(tf, tgt, ctx, top_k=top_k)

        elif method_type == "vanilla_rag":
            out = run_vanilla_rag_edge(tf, tgt, ctx, tok=llm_tok, mdl=llm_mdl, top_k=top_k)

        else:
            raise ValueError(f"Unsupported method_type for classification: {method_type}")

        out["label"] = label
        out["gold_pmids"] = gold_pmids   # << important
        rows.append(out)

    return rows


def run_direct_llm_edge(tf: str, tgt: str, ctx: str, tok, mdl) -> Dict[str, Any]:
    """
    Direct LLM inference for a TF–target–context edge.

    This returns a continuous score derived solely from the LLM probability.
    """
    prompt = rag_mod.build_direct_question_prompt(tf, tgt, ctx)
    llm_score, _, raw = _answer_yes_no_single(tok, mdl, prompt)
    final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(llm_score, [])
    pred_label = int(final_score >= 0.5)
    return {
        "tf": tf,
        "target": tgt,
        "context": ctx,
        "score": float(final_score),
        "pred_label": pred_label,
        "llm_score": float(llm_prob),
        "support_score": float(support_score),
        "raw_answer": raw,
        "retrieved_pmids": [],
    }



def run_ragulate_edge(
    tf: str,
    tgt: str,
    ctx: str,
    tok,
    mdl,
    encoder_name: str,
    top_k: int,
    use_mmr: bool,
    mmr_lambda: float,
) -> Dict[str, Any]:
    
    query = rag_mod.build_query_from_edge(tf, tgt, ctx)
    ret_docs = retrieval.retrieve_vector(
        query,
        encoder_name=encoder_name,
        top_k=top_k,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
    )
    context_text = rag_mod.build_rag_context(tf, tgt, ctx, ret_docs)
    prompt = rag_mod.build_rag_prompt(tf, tgt, ctx, context_text)

    llm_score, _, raw = _answer_yes_no_single(tok, mdl, prompt)
    retrieval_scores = [float(getattr(d, "score", 0.0)) for d in ret_docs] if ret_docs else []
    final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(llm_score, retrieval_scores)
    pred_label = int(final_score >= 0.5)
    return {
        "tf": tf,
        "target": tgt,
        "context": ctx,
        "score": float(final_score),
        "pred_label": pred_label,
        "llm_score": float(llm_prob),
        "support_score": float(support_score),
        "raw_answer": raw,
        "retrieved_pmids": [d.doc_id for d in ret_docs],
    }

def run_hybrid_edge(
    tf: str,
    tgt: str,
    ctx: str,
    tok,
    mdl,
    encoder_name: Optional[str],
    top_k: int,
    use_aliases: bool = False,
    alias_max: Optional[int] = None,
) -> Dict[str, Any]:

    if use_aliases:
        max_aliases = alias_max or getattr(config, "MAX_ALIAS_PER_SYMBOL", 4)

        def _alias_expr(sym: str) -> str:
            exps = alias_mod.expand_symbol_for_query(sym, include_self=True, max_aliases=max_aliases)
            exps = [s for s in exps if s]
            if not exps:
                return sym
            if len(exps) == 1:
                return exps[0]
            return "(" + " OR ".join(exps) + ")"

        tf_expr = _alias_expr(tf)
        tgt_expr = _alias_expr(tgt)
        query = f"{tf_expr} regulates {tgt_expr} in {ctx}"
    else:
        query = rag_mod.build_query_from_edge(tf, tgt, ctx)

    ret_docs = retrieval.retrieve_hybrid_bm25_vector(
        query,
        encoder_name=encoder_name,
        top_k=top_k,
    )

    context_text = rag_mod.build_rag_context(tf, tgt, ctx, ret_docs)
    prompt = rag_mod.build_rag_prompt(tf, tgt, ctx, context_text)
    llm_score, _, raw = _answer_yes_no_single(tok, mdl, prompt)
    retrieval_scores = [float(getattr(d, "score", 0.0)) for d in ret_docs] if ret_docs else []
    final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(llm_score, retrieval_scores)
    pred_label = int(final_score >= 0.5)
    return {
        "tf": tf,
        "target": tgt,
        "context": ctx,
        "score": float(final_score),
        "pred_label": pred_label,
        "llm_score": float(llm_prob),
        "support_score": float(support_score),
        "raw_answer": raw,
        "retrieved_pmids": [d.doc_id for d in ret_docs],
    }


def run_bm25_llm_edge(tf: str, tgt: str, ctx: str, tok, mdl, top_k: int) -> Dict[str, Any]:
    query = rag_mod.build_query_from_edge(tf, tgt, ctx)
    ret_docs = retrieval.retrieve_bm25(query, top_k=top_k)
    context_text = rag_mod.build_rag_context(tf, tgt, ctx, ret_docs)
    prompt = rag_mod.build_rag_prompt(tf, tgt, ctx, context_text)
    llm_score, _, raw = _answer_yes_no_single(tok, mdl, prompt)
    retrieval_scores = [float(getattr(d, "score", 0.0)) for d in ret_docs] if ret_docs else []
    final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(llm_score, retrieval_scores)
    pred_label = int(final_score >= 0.5)
    return {
        "tf": tf,
        "target": tgt,
        "context": ctx,
        "score": float(final_score),
        "pred_label": pred_label,
        "llm_score": float(llm_prob),
        "support_score": float(support_score),
        "raw_answer": raw,
        "retrieved_pmids": [d.doc_id for d in ret_docs],
    }


def run_bm25_simple_edge(tf: str, tgt: str, ctx: str, top_k: int) -> Dict[str, Any]:
    query = rag_mod.build_query_from_edge(tf, tgt, ctx)
    ret_docs = retrieval.retrieve_bm25(query, top_k=top_k)
    retrieval_scores = [float(getattr(d, "score", 0.0)) for d in ret_docs] if ret_docs else []
    final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(0.0, retrieval_scores)
    pred_label = int(final_score >= 0.5)
    raw_answer = f"heuristic: {'supports' if pred_label else 'no evidence'}"
    return {
        "tf": tf,
        "target": tgt,
        "context": ctx,
        "score": float(final_score),
        "pred_label": pred_label,
        "llm_score": float(llm_prob),
        "support_score": float(support_score),
        "raw_answer": raw_answer,
        "retrieved_pmids": [d.doc_id for d in ret_docs],
    }



def run_vanilla_rag_edge(
    tf: str,
    tgt: str,
    ctx: str,
    tok,
    mdl,
    top_k: int,
) -> Dict[str, Any]:
    """
    'Vanilla RAG' classification baseline.

    Uses whatever is implemented in `retrieval.retrieve_vanilla_rag`
    (e.g. a LlamaIndex-based retriever) to fetch top_k documents, then
    builds a RAG context and asks the LLM a yes/no question using the
    same yes/no interface as other methods.

    This keeps the only difference at the *retrieval* layer, so you can
    compare RAGulate vs. vanilla RAG cleanly.
    """
    if tok is None or mdl is None:
        raise ValueError("LLM not initialised for vanilla RAG")

    # 1) Build query and retrieve docs via the vanilla RAG retriever
    query = rag_mod.build_query_from_edge(tf, tgt, ctx)
    ret_docs = retrieval.retrieve_vanilla_rag(query, top_k=top_k)

    # 2) Build RAG context from the retrieved docs
    context_text = rag_mod.build_rag_context(tf, tgt, ctx, ret_docs)

    # 3) Build prompt and query the LLM
    prompt = rag_mod.build_rag_prompt(tf, tgt, ctx, context_text)
    llm_score, _, raw = _answer_yes_no_single(tok, mdl, prompt)
    retrieval_scores = [float(getattr(d, "score", 0.0)) for d in ret_docs] if ret_docs else []
    final_score, llm_prob, support_score = _combine_llm_and_retrieval_scores(llm_score, retrieval_scores)
    pred_label = int(final_score >= 0.5)
    return {
        "tf": tf,
        "target": tgt,
        "context": ctx,
        "score": float(final_score),
        "pred_label": pred_label,
        "llm_score": float(llm_prob),
        "support_score": float(support_score),
        "raw_answer": raw,
        "retrieved_pmids": [d.doc_id for d in ret_docs],
    }



# ---------- Original benchmark pipeline (unchanged) ----------


def _maybe_eval_pmids_field(val: Any) -> Optional[List[str]]:
    """Safely interpret the PMIDs field from the gold dataframe."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return None

def compute_substring_upper_bound(
    gold_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Estimate an abstract-based upper bound on retrieval recall.

    We do this ONLY on positive edges (label == 1) that actually have
    at least one gold PMID. For each such edge, we ask:

      "Does at least one of its gold abstracts contain BOTH the TF
       string and the target string (case-insensitive substring)?"

    The upper_bound_recall is then:
        (# positives where at least one gold abstract has both) /
        (# positives with at least one gold PMID)

    Returns
    -------
    {
      'upper_bound_recall': float,
      'per_edge_has_both': List[bool],        # one per positive edge with gold pmids
      'n_edges_with_gold': int,               # number of such edges
    }
    """
    from .embedding import _load_cached_pubmed_json, _concat_title_abs

    # 1) Restrict to positive edges only
    if "label" not in gold_df.columns:
        raise ValueError("gold_df must contain a 'label' column for positives/negatives")

    pos_df = gold_df[gold_df["label"] == 1].copy()

    if "pmids" not in pos_df.columns:
        raise ValueError("gold_df must contain a 'pmids' column with gold references")

    # 2) Parse pmids field safely
    def _parse_pmids(val: Any) -> List[str]:
        pm_list = _maybe_eval_pmids_field(val)
        if pm_list is None:
            return []
        return [str(p) for p in pm_list if p]

    pos_df["pmid_list"] = pos_df["pmids"].apply(_parse_pmids)

    # Only keep positives that actually have at least one PMID
    pos_with_gold = pos_df[pos_df["pmid_list"].map(len) > 0].copy()
    if pos_with_gold.empty:
        return {
            "upper_bound_recall": 0.0,
            "per_edge_has_both": [],
            "n_edges_with_gold": 0,
        }

    flags: List[bool] = []

    for _, row in pos_with_gold.iterrows():
        tf = str(row["tf"]).strip()
        tgt = str(row["target"]).strip()
        pmids = row["pmid_list"]

        tf_l = tf.lower()
        tgt_l = tgt.lower()

        has_both = False
        for pmid in pmids:
            try:
                rec = _load_cached_pubmed_json(str(pmid))
                txt = _concat_title_abs(rec).lower()
            except Exception:
                continue

            if tf_l in txt and tgt_l in txt:
                has_both = True
                break

        flags.append(has_both)

    n_edges_with_gold = len(flags)
    upper_bound = float(sum(flags)) / float(n_edges_with_gold) if n_edges_with_gold > 0 else 0.0

    return {
        "upper_bound_recall": upper_bound,
        "per_edge_has_both": flags,
        "n_edges_with_gold": n_edges_with_gold,
    }


def compute_substring_upper_bound_precision(
    gold_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Estimate an abstract-based upper bound on retrieval precision.

    We treat a very simple heuristic:
      "An edge is predicted positive by a substring-based method if at
       least one of its gold PMIDs has an abstract that contains BOTH
       the TF symbol and the target symbol (case-insensitive substring)."

    Using this edge-level prediction:
      - TP_edges = edges with label == 1 and predicted positive
      - FP_edges = edges with label == 0 and predicted positive

    Then:
        upper_bound_precision = TP_edges / (TP_edges + FP_edges)

    This gives a *theoretical* precision ceiling for any method that is
    constrained to fire only when TF and target co-occur as substrings
    in the abstract text of PMIDs present in the gold_df.
    """
    from .embedding import _load_cached_pubmed_json, _concat_title_abs

    if "label" not in gold_df.columns:
        raise ValueError("gold_df must contain a 'label' column for positives/negatives")
    if "pmids" not in gold_df.columns:
        raise ValueError("gold_df must contain a 'pmids' column with gold references")

    def _parse_pmids(val: Any) -> List[str]:
        pm_list = _maybe_eval_pmids_field(val)
        if pm_list is None:
            return []
        return [str(p) for p in pm_list if p]

    # Only edges that actually have at least one gold PMID
    df = gold_df.copy()
    df["pmid_list"] = df["pmids"].apply(_parse_pmids)
    df = df[df["pmid_list"].map(len) > 0].copy()

    if df.empty:
        return {
            "upper_bound_precision": 0.0,
            "per_edge_pred_pos": [],
            "n_edges_with_gold": 0,
            "n_tp_edges": 0,
            "n_fp_edges": 0,
        }

    pred_pos_flags: List[bool] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        tf = str(row["tf"]).strip()
        tgt = str(row["target"]).strip()
        pmids = row["pmid_list"]

        tf_l = tf.lower()
        tgt_l = tgt.lower()

        # "Predicted positive" if ANY gold PMID has both TF and target substrings
        has_both = False
        for pmid in pmids:
            try:
                rec = _load_cached_pubmed_json(str(pmid))
                txt = _concat_title_abs(rec).lower()
            except Exception:
                continue
            if tf_l in txt and tgt_l in txt:
                has_both = True
                break

        pred_pos_flags.append(has_both)
        labels.append(int(row["label"]))

    tp_edges = sum(1 for flag, lab in zip(pred_pos_flags, labels) if flag and lab == 1)
    fp_edges = sum(1 for flag, lab in zip(pred_pos_flags, labels) if flag and lab == 0)
    n_edges_with_gold = len(pred_pos_flags)

    denom = tp_edges + fp_edges
    upper_bound_precision = float(tp_edges) / float(denom) if denom > 0 else 0.0

    return {
        "upper_bound_precision": upper_bound_precision,
        "per_edge_pred_pos": pred_pos_flags,
        "n_edges_with_gold": n_edges_with_gold,
        "n_tp_edges": tp_edges,
        "n_fp_edges": fp_edges,
    }


def run_benchmark(
    gold_df: pd.DataFrame,
    top_k: int = config.TOP_K_RETRIEVE,
    use_mistral: Optional[bool] = None,
    use_llama31: Optional[bool] = None,
    use_phi3: Optional[bool] = None,
    use_qwen25: Optional[bool] = None,
    do_permute: bool = False,
    n_perm: int = 0,
) -> Dict[str, Any]:
    """Run the benchmark on a DataFrame of gold edges.

    Parameters mirror those of the original notebook's ``run_benchmark``.
    Returns a dictionary with two keys: ``rows`` containing raw
    per-example outputs and ``metrics`` containing summarised scores.
    """
    # Validate columns
    needed = {"tf", "target", "context", "label"}
    if not needed.issubset(set(gold_df.columns)):
        raise ValueError(f"gold_df must contain columns {needed}")
    # Determine which models to use (fallback to config)
    use_mistral = config.USE_MISTRAL if use_mistral is None else bool(use_mistral)
    use_llama31 = config.USE_LLAMA31 if use_llama31 is None else bool(use_llama31)
    use_phi3 = config.USE_PHI3 if use_phi3 is None else bool(use_phi3)
    use_qwen25 = config.USE_QWEN25 if use_qwen25 is None else bool(use_qwen25)

    N = len(gold_df)
    print(
        f"[benchmark] start: N={N}, top_k={top_k}, models: "
        f"mistral={use_mistral}, llama31={use_llama31}, "
        f"phi3={use_phi3}, qwen25={use_qwen25}"
    )

    # Initialise retriever and models
    retr = get_retriever_safe(top_k=top_k, verbose=True)
    tok_bgpt, mdl_bgpt = get_biogpt()
    tok_mis, mdl_mis = get_mistral() if use_mistral else (None, None)
    tok_ll3, mdl_ll3 = get_llama31() if use_llama31 else (None, None)
    tok_phi, mdl_phi = get_phi3() if use_phi3 else (None, None)
    tok_qw, mdl_qw = get_qwen25() if use_qwen25 else (None, None)

    # Precompute retrieval and RAG contexts
    pre_rows: List[Dict[str, Any]] = []
    labs: List[int] = []
    for _, row in gold_df.iterrows():
        tf = row["tf"]
        tg = row["target"]
        ctx = row["context"]
        lab = int(row["label"])
        labs.append(lab)
        query = f"{tf} regulates {tg} in {ctx}"
        pmids_field = (
            _maybe_eval_pmids_field(row.get("pmids")) if "pmids" in row else None
        )
        if pmids_field is None:
            hits = retr.retrieve(query, top_k=top_k)
        else:
            hits = retr.retrieve(query, top_k=top_k, candidate_pmids=pmids_field)
        rag_ctx = _get_rag_ctx_cached(
            query, hits, max_sent=config.MAX_SENT_PER_PAPER
        )
        pre_rows.append({"tf": tf, "tg": tg, "ctx": ctx, "lab": lab, "rag": rag_ctx})

    # Helper to construct prompts
    def _prompts(rows: List[Dict[str, Any]], tok, mdl, with_rag: bool) -> List[str]:
        ps: List[str] = []
        for r in rows:
            base = _prompt_template(r["tf"], r["tg"], r["ctx"])
            rag_ctx = r["rag"] if with_rag else None
            ps.append(
                build_prompt_with_budget(
                    tok,
                    mdl,
                    base,
                    rag_ctx,
                    max_new_tokens=config.GEN_MAX_NEW,
                    safety_margin=config.SAFETY_MARGIN,
                )
            )
        return ps

    # Helper to run and convert raw text outputs to scores
    def _run_and_score(tok, mdl, with_rag: bool) -> List[Optional[float]]:
        if tok is None or mdl is None:
            return [None] * len(pre_rows)
        prompts = _prompts(pre_rows, tok, mdl, with_rag=with_rag)
        txts = _llm_generate_batch(tok, mdl, prompts, max_new_tokens=config.GEN_MAX_NEW)
        return [float(_binary_from_text(t)) for t in txts]

    # Run each model (with and without RAG)
    def _maybe_print(mdl, name: str) -> None:
        if mdl is not None:
            print(name, flush=True)

    _maybe_print(mdl_bgpt, "BioGPT")
    s_bgpt_only = _run_and_score(tok_bgpt, mdl_bgpt, with_rag=False)
    _maybe_print(mdl_bgpt, "BioGPT +RAG")
    s_bgpt_rag = _run_and_score(tok_bgpt, mdl_bgpt, with_rag=True)

    _maybe_print(mdl_mis, "Mistral")
    s_mis_only = _run_and_score(tok_mis, mdl_mis, with_rag=False)
    _maybe_print(mdl_mis, "Mistral +RAG")
    s_mis_rag = _run_and_score(tok_mis, mdl_mis, with_rag=True)

    _maybe_print(mdl_ll3, "Llama3.1")
    s_ll3_only = _run_and_score(tok_ll3, mdl_ll3, with_rag=False)
    _maybe_print(mdl_ll3, "Llama3.1 +RAG")
    s_ll3_rag = _run_and_score(tok_ll3, mdl_ll3, with_rag=True)

    _maybe_print(mdl_phi, "Phi-3")
    s_phi_only = _run_and_score(tok_phi, mdl_phi, with_rag=False)
    _maybe_print(mdl_phi, "Phi-3 +RAG")
    s_phi_rag = _run_and_score(tok_phi, mdl_phi, with_rag=True)

    _maybe_print(mdl_qw, "Qwen2.5")
    s_qw_only = _run_and_score(tok_qw, mdl_qw, with_rag=False)
    _maybe_print(mdl_qw, "Qwen2.5 +RAG")
    s_qw_rag = _run_and_score(tok_qw, mdl_qw, with_rag=True)

    # Baselines: global cosine and context keyword match
    def score_global_cosine(
        tf: str,
        tg: str,
        ctx: str,
        retriever=None,
        top_k: int = config.TOP_K_RETRIEVE,
    ) -> float:
        retr2 = retriever or get_retriever_safe(top_k=top_k, verbose=False)
        query = f"{tf} regulates {tg} in {ctx}"
        hits2 = retr2.retrieve(query, top_k=1)
        return 0.0 if not hits2 else float(max(h["score"] for h in hits2))

    def score_ctx_only_keyword(tf: str, tg: str, ctx: str) -> float:
        ctx_l = (ctx or "").lower()
        tf_l = (tf or "").lower()
        tg_l = (tg or "").lower()
        ok = (tf_l in ctx_l) or (tg_l in ctx_l)
        return 1.0 if ok else 0.0

    # Additional baseline: BioGPT support prompt (rag context only)
    prompts_support = _prompts(pre_rows, tok_bgpt, mdl_bgpt, with_rag=True)
    txt_support = _llm_generate_batch(
        tok_bgpt, mdl_bgpt, prompts_support, max_new_tokens=config.GEN_MAX_NEW
    )
    s_support = [float(_binary_from_text(t)) for t in txt_support]

    # Assemble row outputs
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(pre_rows):
        outrow = {
            "tf": r["tf"],
            "target": r["tg"],
            "context": r["ctx"],
            "label": r["lab"],
        }
        outrow.update(
            {
                "biogpt_only": s_bgpt_only[i],
                "biogpt_rag": s_bgpt_rag[i],
                "mistral_only": s_mis_only[i],
                "mistral_rag": s_mis_rag[i],
                "llama31_only": s_ll3_only[i],
                "llama31_rag": s_ll3_rag[i],
                "phi3_only": s_phi_only[i],
                "phi3_rag": s_phi_rag[i],
                "qwen25_only": s_qw_only[i],
                "qwen25_rag": s_qw_rag[i],
                "global_cosine": score_global_cosine(
                    r["tf"], r["tg"], r["ctx"], retriever=retr, top_k=top_k
                ),
                "ctx_only_keyword": score_ctx_only_keyword(
                    r["tf"], r["tg"], r["ctx"]
                ),
                "support_only": s_support[i],
            }
        )
        rows.append(outrow)

    # Compute metrics per method
    metrics: Dict[str, Dict[str, Any]] = {}
    labs_int = [r["label"] for r in rows]
    methods = [
        "biogpt_only",
        "biogpt_rag",
        "mistral_only",
        "mistral_rag",
        "llama31_only",
        "llama31_rag",
        "phi3_only",
        "phi3_rag",
        "qwen25_only",
        "qwen25_rag",
        "global_cosine",
        "ctx_only_keyword",
        "support_only",
    ]
    for m in methods:
        vals = [row[m] for row in rows if row[m] is not None]
        if len(vals) == len(rows):
            met = evaluate_binary(labs_int, vals)
            if do_permute and n_perm > 0 and met.get("auroc") is not None:
                perm = permutation_test(
                    labs_int, vals, n_perm=n_perm, metric="auroc", seed=config.SEED
                )
                if perm is not None:
                    met["perm_auroc"] = perm
            if do_permute and n_perm > 0 and met.get("auprc") is not None:
                perm = permutation_test(
                    labs_int,
                    vals,
                    n_perm=n_perm,
                    metric="auprc",
                    seed=config.SEED + 1,
                )
                if perm is not None:
                    met["perm_auprc"] = perm
            metrics[m] = met

    # Print some diagnostics
    def _nn(xs: List[Optional[float]]) -> int:
        return sum(1 for v in xs if v is not None)

    summary = {k: 0 for k in methods}
    for k in summary.keys():
        summary[k] = _nn([r[k] for r in rows])
    print("[ran] counts per method:", summary)
    N_rows = len(rows)
    assert summary["biogpt_only"] == N_rows and summary["biogpt_rag"] == N_rows
    if use_mistral:
        assert summary["mistral_only"] == N_rows and summary["mistral_rag"] == N_rows
    if use_llama31:
        assert summary["llama31_only"] == N_rows and summary["llama31_rag"] == N_rows
    if use_phi3:
        assert summary["phi3_only"] == N_rows and summary["phi3_rag"] == N_rows
    if use_qwen25:
        assert summary["qwen25_only"] == N_rows and summary["qwen25_rag"] == N_rows
    assert (
        summary["global_cosine"] == N_rows
        and summary["ctx_only_keyword"] == N_rows
        and summary["support_only"] == N_rows
    )
    print("[ok] benchmark completed for all enabled models")

    return {"rows": rows, "metrics": metrics}


def run_benchmark_epochs(
    gold_df: pd.DataFrame,
    epochs: int = config.EPOCHS,
    top_k: int = config.TOP_K_RETRIEVE,
    use_mistral: Optional[bool] = None,
    use_llama31: Optional[bool] = None,
    use_phi3: Optional[bool] = None,
    use_qwen25: Optional[bool] = None,
    do_permute: bool = False,
    n_perm: int = 0,
) -> List[Dict[str, Any]]:
    """Run the benchmark for multiple epochs and collect history."""
    print_config_banner(len(gold_df), epochs)
    history: List[Dict[str, Any]] = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        out = run_benchmark(
            gold_df,
            top_k=top_k,
            use_mistral=use_mistral,
            use_llama31=use_llama31,
            use_phi3=use_phi3,
            use_qwen25=use_qwen25,
            do_permute=do_permute,
            n_perm=n_perm,
        )
        history.append({"epoch": ep, "metrics": out["metrics"]})
        elapsed = time.time() - t0
        parts: List[str] = []
        for k, v in out["metrics"].items():
            if v and isinstance(v, dict) and v.get("auroc") is not None:
                parts.append(f"{k}: AUROC={v['auroc']:.3f} AUPRC={v['auprc']:.3f}")
        print(f"[epoch {ep}/{epochs}] {', '.join(parts)} | {elapsed:.1f}s")
    return history


def save_benchmark(
    outputs: Dict[str, Any],
    rows_csv: str = str(config.ROWS_CSV),
    metrics_json: str = str(config.METRICS_JSON),
) -> None:
    """Persist benchmark results to disk."""
    rows = outputs.get("rows", [])
    metrics = outputs.get("metrics", {})
    if rows:
        pd.DataFrame(rows).to_csv(rows_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        import json as _json

        _json.dump(metrics, f, indent=2)
    if config.VERBOSE >= 1:
        print(f"[ok] wrote {rows_csv} and {metrics_json}")


def print_metrics_table(metrics: Dict[str, Dict[str, Any]]) -> None:
    """Nicely print a metrics summary table to stdout."""
    print("\n[metrics]")
    for m, vals in metrics.items():
        if not vals:
            continue
        auroc = vals.get("auroc")
        auprc = vals.get("auprc")
        f1 = vals.get("best_f1")
        s_auroc = "None" if auroc is None else f"{auroc:.3f}"
        s_auprc = "None" if auprc is None else f"{auprc:.3f}"
        s_f1 = "None" if f1 is None else f"{f1:.3f}"
        line = f"  {m:18s} AUROC={s_auroc}  AUPRC={s_auprc}  F1*={s_f1}"
        if "perm_auroc" in vals:
            pa = vals["perm_auroc"]
            line += (
                f" | perm AUROC p={pa['p_value']:.3f} "
                f"CI=({pa['null_ci'][0]:.3f},{pa['null_ci'][1]:.3f}) "
                f"n={pa['n_perm']}"
            )
        if "perm_auprc" in vals:
            pp = vals["perm_auprc"]
            line += (
                f" | perm AUPRC p={pp['p_value']:.3f} "
                f"CI=({pp['null_ci'][0]:.3f},{pp['null_ci'][1]:.3f}) "
                f"n={pp['n_perm']}"
            )
        print(line)


__all__ = [
    "run_benchmark",
    "run_benchmark_epochs",
    "save_benchmark",
    "print_metrics_table",
    "run_retrieval_experiment",
    "run_classification_experiment",
    "run_direct_llm_edge",
    "run_ragulate_edge",
    "run_hybrid_edge",
    "run_bm25_llm_edge",
    "run_bm25_simple_edge",
    "run_vanilla_rag_edge",
    "run_ragulate_inference",
    "run_llm_only_inference",          # <-- new
    "compute_substring_upper_bound",
    "compute_substring_upper_bound_precision",
    "save_ragulate_results_csv",
    "save_ragulate_results_html",
    "evaluate_pmid_hallucination",
    "run_pmid_hallucination_suite",
]

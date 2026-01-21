"""
RAG context and prompt construction utilities.

This module contains functions that select salient sentences
from retrieved documents, build RAG contexts, and assemble
prompts suitable for feeding into language models. It also
provides helpers for interpreting binary yes/no answers.
"""

from typing import List, Optional, Tuple, Iterable, Any
import re
import numpy as np

from . import config
from .state import GLOBAL_EMB, GLOBAL_VECS, _RAG_CTX_CACHE, _SENT_EMB_CACHE
from .retrieval import get_retriever_safe
from .llm_models import _model_max_ctx

# Regular expression used for naive sentence splitting.
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# ----------------------------------------------------------------------
# Optional: simple context keyword expansion for retrieval queries.
# This mirrors the context categories used in the gold standard.
# ----------------------------------------------------------------------
_CONTEXT_KW = {
    "immune": [
        "immune", "b cell", "t cell", "lymphocyte", "macrophage",
        "monocyte", "dendritic", "cytokine", "chemokine", "mhc",
        "antigen", "immunology"
    ],
    "neural": [
        "neural", "neuron", "brain", "hippocampus", "cortex",
        "astrocyte", "glia", "oligodendrocyte", "microglia", "synapse"
    ],
    "cardiac": [
        "cardiac", "heart", "cardiomyocyte", "myocardium",
        "ventricle", "atrium"
    ],
    "hepatic": [
        "hepatic", "liver", "hepatocyte", "kupffer", "bile"
    ],
    "renal": [
        "renal", "kidney", "nephron", "glomerulus", "podocyte"
    ],
    "muscle": [
        "muscle", "skeletal muscle", "myoblast", "myotube",
        "myofiber", "myogenesis", "sarcomere"
    ],
    "epithelial": [
        "epithelial", "epithelium", "keratinocyte", "epidermis",
        "mucosa"
    ],
    "stem": [
        "stem cell", "pluripotent", "ipsc", "progenitor",
        "hematopoietic stem", "mesenchymal stem"
    ],
}


def _expand_context_terms(context: str, max_terms: int = 4) -> List[str]:
    """
    Given a canonical context label (e.g. 'muscle', 'hepatic'),
    return a small list of extra keywords for retrieval.
    """
    ctx = (context or "").strip().lower()
    if not ctx:
        return []
    kw = _CONTEXT_KW.get(ctx, [])
    if not kw:
        return []
    return kw[: max(0, int(max_terms))]

def truncate_text_to_tokens(tok, text: str, max_tokens: int) -> str:
    """Truncate a string so that it encodes to no more than ``max_tokens`` tokens."""
    if max_tokens <= 0 or not text:
        return ''
    enc = tok(text, add_special_tokens=False)
    ids = enc['input_ids'][:max_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def build_prompt_with_budget(
    tok,
    mdl,
    base_prompt: str,
    rag_context: Optional[str],
    max_new_tokens: int = config.GEN_MAX_NEW,
    safety_margin: int = config.SAFETY_MARGIN,
) -> str:
    """Construct a prompt that fits within the model's context window."""
    max_ctx = _model_max_ctx(tok, mdl)
    budget = max(8, max_ctx - max_new_tokens - safety_margin)
    base_ids = tok(base_prompt, add_special_tokens=False)['input_ids']
    if len(base_ids) > budget:
        base_prompt = truncate_text_to_tokens(tok, base_prompt, budget)
    remaining = max(
        0,
        budget - len(tok(base_prompt, add_special_tokens=False)['input_ids']),
    )
    if rag_context:
        rag_trim = truncate_text_to_tokens(tok, rag_context, remaining)
        prompt = f"{base_prompt}\n\nEvidence:\n{rag_trim}\n\nAnswer:"
    else:
        prompt = f"{base_prompt}\n\nAnswer:"
    return prompt

def build_query_from_edge(
    tf: str,
    target: str,
    context: str,
    mode: str = "tf_tg_ctx",
    expand_context: bool = False,
    max_ctx_terms: int = 4,
) -> str:
    """
    Canonical text query used for retrieval from a (TF, target, context) triple.

    Parameters
    ----------
    tf : str
        Transcription factor symbol/name.
    target : str
        Target gene symbol/name.
    context : str
        Canonical context label (e.g. 'muscle', 'hepatic', ...).
    mode : {'tf_tg_ctx', 'tf_tg_only'}
        - 'tf_tg_ctx': include TF, target, and context in the query.
        - 'tf_tg_only': ignore context and use only TF + target.
    expand_context : bool
        If True, augment the context with a few extra keywords
        based on a simple dictionary (_CONTEXT_KW).
    max_ctx_terms : int
        Maximum number of context keywords to inject when expand_context is True.
    """
    tf = (tf or "").strip()
    target = (target or "").strip()
    context = (context or "").strip()

    # Fallbacks if one of TF / target is missing
    if not tf and not target and not context:
        return ""
    if not tf and target:
        base = target
    elif tf and not target:
        base = tf
    else:
        base = f"{tf} regulates {target}"

    # Mode: TF + target only (ignore context)
    if mode == "tf_tg_only":
        return base

    # Default: TF + target + context
    if not context:
        # No context string given -> just return base
        return base

    # Optional context expansion
    if expand_context:
        extra_terms = _expand_context_terms(context, max_terms=max_ctx_terms)
        if extra_terms:
            ctx_phrase = " ".join(extra_terms)
            # Example: "TBP regulates MYOG in muscle (muscle skeletal muscle myoblast myotube)"
            return f"{base} in {context} ({ctx_phrase})"
        else:
            return f"{base} in {context}"
    else:
        return f"{base} in {context}"





def _prompt_template(tf: str, tg: str, ctx: str) -> str:
    """
    Return the base prompt before RAG context is inserted.

    This version explicitly frames the model as a strict binary classifier
    and emphasises that the answer must be a single 'yes'/'no' token with
    no explanations or extra words.
    """
    return (
        "You are a strict binary classifier.\n\n"
        "Task:\n"
        "Determine whether there is direct regulatory evidence that a "
        "transcription factor regulates a target gene in the given biological "
        "context.\n\n"
        "You MUST answer with exactly one token:\n"
        '- \"yes\"\n'
        '- \"no\"\n\n'
        "NO explanations. NO punctuation. NO extra words.\n\n"
        f"TF: {tf}\n"
        f"Target gene: {tg}\n"
        f"Biological context: {ctx}"
    )


def build_direct_question_prompt(tf: str, tgt: str, ctx: str) -> str:
    return _prompt_template(tf, tgt, ctx)


def _split_sentences(txt: str) -> List[str]:
    """Split a block of text into sentences using a simple regex."""
    ss = [s.strip() for s in SENT_SPLIT_RE.split(txt or '')]
    return [s for s in ss if s]


def _sent_embed(sent_list: List[str]) -> np.ndarray:
    """Embed a list of sentences using the global embedding model."""
    if not sent_list:
        return np.zeros((0, GLOBAL_VECS.shape[1]), np.float32)
    if GLOBAL_EMB is None:
        raise RuntimeError('Global sentence embedding model has not been initialised')
    return GLOBAL_EMB.encode(
        sent_list,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def _select_relevant_sentences(
    text: str,
    query: str,
    max_sent: int = config.MAX_SENT_PER_PAPER,
) -> List[str]:
    """Choose the most relevant sentences from a document for a given query."""
    if not text:
        return []
    sents = _split_sentences(text)
    if len(sents) <= max_sent:
        return sents
    key = tuple(sents)
    vecs = _SENT_EMB_CACHE.get(key)
    if vecs is None:
        vecs = _sent_embed(sents)
        _SENT_EMB_CACHE[key] = vecs
    qv = GLOBAL_EMB.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    sims = vecs @ qv
    top = np.argpartition(-sims, max_sent - 1)[:max_sent]
    top = top[np.argsort(-sims[top])]
    return [sents[i] for i in top]


def _build_rag_context(
    query: str,
    hits: List[dict],
    max_sent: int = config.MAX_SENT_PER_PAPER,
) -> str:
    """Concatenate relevant sentences from retrieved documents into a RAG context."""
    from .embedding import _load_cached_pubmed_json, _concat_title_abs  # local import
    blocks: List[str] = []
    for h in hits:
        pmid = h['pmid']
        try:
            rec = _load_cached_pubmed_json(pmid)
            txt = _concat_title_abs(rec)
            sents = _select_relevant_sentences(txt, query, max_sent=max_sent)
            if sents:
                blocks.append(f"[PMID {pmid}] " + ' '.join(sents))
            if len(blocks) >= config.MAX_PMIDS_PER_EDGE:
                break
        except Exception:
            continue
    return '\n\n'.join(blocks)


def _get_rag_ctx_cached(
    query: str,
    hits: List[dict],
    max_sent: int = config.MAX_SENT_PER_PAPER,
) -> str:
    """Cache RAG contexts keyed by the query and hit list."""
    key = (query, tuple(h['pmid'] for h in hits), int(max_sent))
    cached = _RAG_CTX_CACHE.get(key)
    if cached is not None:
        return cached
    ctx = _build_rag_context(query, hits, max_sent=max_sent)
    _RAG_CTX_CACHE[key] = ctx
    return ctx


def _binary_from_text(t: str) -> float:
    """
    Map free-form text to a binary label.

    Priority:
      1) First word (most reliable for instruct models)
      2) Then presence of 'yes' or 'no' anywhere
      3) Otherwise 0.5 (uncertain)
    """
    t = (t or "").strip().lower()

    # Strip leading punctuation / tags
    for ch in " .,:;\"'()[]{}<>":
        t = t.lstrip(ch).lstrip()

    if not t:
        return 0.5

    first_word = t.split()[0]

    if first_word.startswith("yes"):
        return 1.0
    if first_word.startswith("no"):
        return 0.0

    # Fallback heuristics
    if "yes" in t and "no" not in t:
        return 1.0
    if "no" in t and "yes" not in t:
        return 0.0

    return 0.5


def build_rag_context(
    tf: str,
    tgt: str,
    ctx: str,
    ret_docs: List[Any],
    max_sent: int = config.MAX_SENT_PER_PAPER,
) -> str:
    """
    Public wrapper to build a RAG context from retrieved documents.
    """
    query = build_query_from_edge(tf, tgt, ctx)
    hits = [
        {"pmid": d.doc_id, "score": float(getattr(d, "score", 0.0))}
        for d in ret_docs
    ]
    return _get_rag_ctx_cached(query, hits, max_sent=max_sent)


def build_rag_prompt(
    tf: str,
    tgt: str,
    ctx: str,
    rag_context: Optional[str],
) -> str:
    """
    Build the final RAG prompt string from an edge and context.
    """
    base = _prompt_template(tf, tgt, ctx)
    if rag_context:
        return f"{base}\n\nEvidence:\n{rag_context}\n\nAnswer:"
    return f"{base}\n\nAnswer:"

def support_score_from_scores(
    scores: Iterable[float],
    method: str = "max",
    temperature: float = 1.0,
) -> float:
    """
    Aggregate retrieval scores into a single support score in [0, 1].

    - method="max": use the maximum retrieval score.
    - method="mean": use the mean retrieval score.
    - temperature: scales how steep the logistic mapping is.
    """
    import numpy as _np

    arr = _np.array(list(scores), dtype=float)
    if arr.size == 0:
        return 0.0

    if method == "mean":
        agg = float(arr.mean())
    else:
        agg = float(arr.max())

    temp = float(temperature) if temperature and temperature > 0 else 1.0
    x = agg / temp
    x = float(max(min(x, 50.0), -50.0))  # avoid overflow
    return float(1.0 / (1.0 + _np.exp(-x)))


__all__ = [
    "truncate_text_to_tokens",
    "build_prompt_with_budget",
    "build_query_from_edge",
    "_split_sentences",
    "_select_relevant_sentences",
    "_build_rag_context",
    "_get_rag_ctx_cached",
    "_prompt_template",
    "_binary_from_text",
    "build_direct_question_prompt",
    "build_rag_context",
    "build_rag_prompt",
    "support_score_from_scores", 
]

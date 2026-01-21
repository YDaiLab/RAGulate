"""
Document retrieval logic for the RAGulate benchmark.

This module implements vector-based search over the precomputed
PubMed corpus, a BM25 backend, and a "vanilla RAG" backend via
LlamaIndex.

We support multiple encoders (MiniLM vs BioBERT) and optional
MMR re-ranking, while keeping a backwards-compatible API for
the rest of the pipeline.
"""

from typing import List, Optional, Dict, Any, Tuple
import os
import re

import numpy as np

from rank_bm25 import BM25Okapi  # real BM25

from . import state
from . import config
from .embedding import (
    get_sentence_embedder,
    _ensure_global_corpus,
    _load_cached_pubmed_json,
    _concat_title_abs,
)

from . import llamaindex_utils


# ---------------------------------------------------------------------
# Internal registries
# ---------------------------------------------------------------------

# Per-encoder corpus: encoder_name -> {"pmids": List[str], "vecs": np.ndarray}
# encoder_name is ALWAYS the canonical alias: "minilm" or "biobert"
_ENCODER_CORPORA: Dict[str, Dict[str, Any]] = {}

# Per-encoder query encoder (SentenceTransformer or compatible)
_ENCODER_MODELS: Dict[str, Any] = {}

# Retrievers cached by (encoder_name, top_k)
_RETRIEVER_CACHE: Dict[Tuple[str, int], "PubMedRetriever"] = {}

# BM25 index and corpus tokens
_BM25_INDEX: Optional[BM25Okapi] = None
_BM25_PMIDS: List[str] = []
_BM25_TOKENS: List[List[str]] = []


# ---------------------------------------------------------------------
# Canonical encoder naming
# ---------------------------------------------------------------------

def _canonical_encoder_name(encoder_name: Optional[str]) -> str:
    """
    Map various aliases / HF ids to a canonical internal name.

    Returns:
        "minilm" or "biobert"
    """
    enc = (encoder_name or "").strip()

    minilm_model = getattr(
        config,
        "EMBED_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    biobert_sent = getattr(
        config,
        "BIOBERT_SENTENCE_MODEL_NAME",
        getattr(
            config,
            "BIOBERT_EMBED_MODEL_NAME",
            "dmis-lab/biobert-base-cased-v1.1",
        ),
    )

    # MiniLM / default encoder
    if enc in ("", "minilm", "mini_lm", minilm_model):
        return "minilm"

    # BioBERT encoder
    if enc in ("biobert", biobert_sent):
        return "biobert"

    raise ValueError(f"Unsupported encoder_name: {encoder_name!r}")


def _default_minilm_name() -> str:
    """Canonical name for the MiniLM encoder."""
    return "minilm"


def _biobert_name() -> str:
    """Canonical name for the BioBERT encoder."""
    return "biobert"


# ---------------------------------------------------------------------
# Helpers to build / load encoder-specific corpora
# ---------------------------------------------------------------------

def _ensure_minilm_corpus(verbose: bool = True) -> None:
    """
    Ensure that the default MiniLM corpus is built and registered
    under encoder_name="minilm".
    """
    encoder_name = _default_minilm_name()
    if encoder_name in _ENCODER_CORPORA:
        return

    # Build the original global corpus and populate
    # state.GLOBAL_PMIDS / GLOBAL_VECS / GLOBAL_EMB
    _ensure_global_corpus(config.EMBED_MODEL_NAME, verbose=verbose)

    pmids = list(state.GLOBAL_PMIDS)
    vecs = state.GLOBAL_VECS
    emb_model = state.GLOBAL_EMB

    if vecs is None or emb_model is None:
        raise RuntimeError("MiniLM global corpus was not initialised correctly")

    _ENCODER_CORPORA[encoder_name] = {
        "pmids": pmids,
        "vecs": vecs,
    }
    _ENCODER_MODELS[encoder_name] = emb_model


def _biobert_cache_paths() -> Tuple[str, str]:
    """
    Derive file paths for cached BioBERT corpus embeddings.

    We piggy-back on EMB_CACHE_NPZ but store a separate *_biobert.npz
    so that MiniLM and BioBERT corpora do not overwrite each other.
    """
    base = str(config.EMB_CACHE_NPZ)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".npz"
    npz_path = root + "_biobert" + ext

    meta_base = str(
        getattr(
            config,
            "EMB_CACHE_META",
            root + "_biobert_meta.json",
        )
    )
    meta_root, meta_ext = os.path.splitext(meta_base)
    if not meta_ext:
        meta_ext = ".json"
    meta_path = meta_root + meta_ext
    return npz_path, meta_path


def _ensure_biobert_corpus(verbose: bool = True) -> None:
    """
    Ensure that the BioBERT corpus is built and registered under
    encoder_name="biobert".

    If a cached NPZ exists it is loaded; otherwise, title+abstract
    text is embedded with BioBERT and saved for future runs.
    """
    encoder_name = _biobert_name()
    if encoder_name in _ENCODER_CORPORA:
        return

    npz_path, meta_path = _biobert_cache_paths()

    # ------------------------------------------------------------------
    # 1) Try to load cached BioBERT vectors
    # ------------------------------------------------------------------
    if os.path.exists(npz_path):
        if verbose and config.VERBOSE >= 1:
            print(f"[biobert-corpus] loading cached embeddings from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        pmids = list(data["pmids"])
        vecs = np.array(data["vecs"])

        # Load / remember model name if present
        model_name = getattr(config, "BIOBERT_EMBED_MODEL_NAME", None)
        if os.path.exists(meta_path):
            try:
                import json
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                model_name = meta.get("model_name", model_name)
            except Exception:
                pass
        if model_name is None:
            model_name = getattr(
                config,
                "BIOBERT_EMBED_MODEL_NAME",
                "dmis-lab/biobert-base-cased-v1.1",
            )

        emb_model = get_sentence_embedder(model_name)
        # Update the global sentence embedding model so that RAG context
        # construction uses the correct encoder when BioBERT is requested.
        state.GLOBAL_EMB = emb_model
        _ENCODER_CORPORA[encoder_name] = {"pmids": pmids, "vecs": vecs}
        _ENCODER_MODELS[encoder_name] = emb_model
        return

    # ------------------------------------------------------------------
    # 2) Build fresh BioBERT corpus embeddings
    # ------------------------------------------------------------------
    # Reuse the same PMIDs in the same order as MiniLM
    _ensure_minilm_corpus(verbose=verbose)
    base_pmids = _ENCODER_CORPORA[_default_minilm_name()]["pmids"]
    pmids = list(base_pmids)

    model_name = getattr(
        config,
        "BIOBERT_EMBED_MODEL_NAME",
        "dmis-lab/biobert-base-cased-v1.1",
    )
    if verbose and config.VERBOSE >= 1:
        print(f"[biobert-corpus] initialising BioBERT encoder: {model_name}")
    emb_model = get_sentence_embedder(model_name)
    # Update the global embedding model so sentence selection aligns with BioBERT
    state.GLOBAL_EMB = emb_model

    texts: List[str] = []
    for pmid in pmids:
        try:
            rec = _load_cached_pubmed_json(pmid)
            txt = _concat_title_abs(rec)
            if not txt:
                txt = ""
            texts.append(txt)
        except Exception:
            texts.append("")

    if verbose and config.VERBOSE >= 1:
        print(f"[biobert-corpus] embedding {len(pmids)} documents")

    vecs = emb_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if not isinstance(vecs, np.ndarray):
        vecs = np.asarray(vecs)

    if verbose and config.VERBOSE >= 1:
        print(f"[biobert-corpus] built matrix: shape={vecs.shape}")

    # Cache to disk for faster future runs
    try:
        if verbose and config.VERBOSE >= 1:
            print(f"[biobert-corpus] saving cache to {npz_path}")
        np.savez(npz_path, pmids=np.array(pmids, dtype=object), vecs=vecs)
        try:
            import json
            with open(meta_path, "w") as f:
                json.dump({"model_name": model_name}, f)
        except Exception:
            pass
    except Exception as e:
        print(f"[warn] could not save BioBERT embedding cache: {e}")

    _ENCODER_CORPORA[encoder_name] = {"pmids": pmids, "vecs": vecs}
    _ENCODER_MODELS[encoder_name] = emb_model


def _ensure_encoder_corpus(encoder_name: Optional[str], verbose: bool = True) -> None:
    """
    Ensure that the corpus for the given encoder is built and registered
    in _ENCODER_CORPORA / _ENCODER_MODELS.

    Accepts aliases like "minilm" / "biobert" and HF ids.
    """
    enc = _canonical_encoder_name(encoder_name)
    if enc == "minilm":
        _ensure_minilm_corpus(verbose=verbose)
        return
    if enc == "biobert":
        _ensure_biobert_corpus(verbose=verbose)
        return
    # Should not reach here because _canonical_encoder_name would raise
    raise ValueError(f"Unsupported encoder_name after canonicalisation: {encoder_name!r}")


# ---------------------------------------------------------------------
# Core retriever
# ---------------------------------------------------------------------

class PubMedRetriever:
    """
    Vector search over the PubMed corpus for a specific encoder.

    This retriever projects queries into the same embedding space as
    the corpus and computes cosine similarity via a dot product. If
    a list of candidate PMIDs is provided the search is restricted
    to those documents.

    Results are returned as dictionaries with ``pmid`` and ``score``.
    """

    def __init__(self, encoder_name: str, default_top_k: int) -> None:
        # Always store canonical name internally
        self.encoder_name = _canonical_encoder_name(encoder_name)
        self.default_top_k = int(default_top_k)

        # Ensure corresponding corpus & encoder exist
        _ensure_encoder_corpus(self.encoder_name, verbose=False)

        corp = _ENCODER_CORPORA[self.encoder_name]
        self._pmids: List[str] = corp["pmids"]
        self._vecs: np.ndarray = corp["vecs"]
        self._pmid_to_idx: Dict[str, int] = {
            pmid: i for i, pmid in enumerate(self._pmids)
        }
        self._encoder = _ENCODER_MODELS[self.encoder_name]

        # Local query cache (keyed by raw query string)
        self._query_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Low-level utilities
    # ------------------------------------------------------------------

    def _qvec(self, query: str) -> np.ndarray:
        """
        Compute (or look up) the embedding vector for a query string.
        """
        if query in self._query_cache:
            return self._query_cache[query]

        v = self._encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        self._query_cache[query] = v
        return v

    # ------------------------------------------------------------------
    # Main retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_pmids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar documents for a query.

        Parameters
        ----------
        query:
            Natural language query, typically
            "TF regulates target in context".
        top_k:
            Number of results to return. Defaults to ``self.default_top_k``.
        candidate_pmids:
            Optional subset of PMIDs to restrict retrieval to.

        Returns
        -------
        list of dict
            Each element has fields:
              - "pmid": str
              - "score": float cosine similarity
        """
        if self._vecs is None or len(self._pmids) == 0:
            return []

        k = int(top_k) if top_k is not None else self.default_top_k
        if k <= 0:
            return []

        qv = self._qvec(query)

        # No candidate filter: full corpus search
        if candidate_pmids is None:
            sims = self._vecs @ qv
            k = min(k, len(sims))
            if k == 0:
                return []
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            return [
                {"pmid": self._pmids[i], "score": float(sims[i])}
                for i in idx
            ]

        # Restricted search
        cand_idx = [self._pmid_to_idx[p] for p in candidate_pmids if p in self._pmid_to_idx]
        if not cand_idx:
            return []
        sims = self._vecs[cand_idx] @ qv
        k = min(k, len(cand_idx))
        if k == 0:
            return []
        ii = np.argpartition(-sims, k - 1)[:k]
        ii = ii[np.argsort(-sims[ii])]
        return [
            {"pmid": self._pmids[cand_idx[i]], "score": float(sims[ii[i]])}
            for i in range(len(ii))
        ]


# ---------------------------------------------------------------------
# Public factory for retrievers
# ---------------------------------------------------------------------

def ensure_pubmed_retriever(
    encoder_name: Optional[str] = None,
    top_k: int = config.TOP_K_DEFAULT,
    rebuild: bool = False,
    verbose: bool = True,
) -> Tuple[None, PubMedRetriever]:
    """
    Ensure that a :class:`PubMedRetriever` is available for the
    requested encoder and return it (wrapped in a (None, retriever)
    tuple for historical reasons).
    """
    enc = _canonical_encoder_name(encoder_name)

    if rebuild:
        if verbose:
            print("[rebuild] invalidating encoder corpora and global state")
        _ENCODER_CORPORA.clear()
        _ENCODER_MODELS.clear()
        _RETRIEVER_CACHE.clear()
        state.GLOBAL_PMIDS = []
        state.GLOBAL_VECS = None
        state.GLOBAL_EMB = None

    # Build corpus for this encoder if needed
    _ensure_encoder_corpus(enc, verbose=verbose)

    key = (enc, int(top_k))
    if key in _RETRIEVER_CACHE:
        return (None, _RETRIEVER_CACHE[key])

    if verbose and config.VERBOSE >= 1:
        print(f"[retriever] create new instance (encoder={enc}, topk={top_k})")

    retr = PubMedRetriever(
        encoder_name=enc,
        default_top_k=top_k,
    )
    _RETRIEVER_CACHE[key] = retr
    return (None, retr)


def get_retriever_safe(
    encoder_name: Optional[str] = None,
    top_k: int = config.TOP_K_DEFAULT,
    verbose: bool = False,
) -> PubMedRetriever:
    """
    Shortcut for obtaining a cached retriever without exposing the
    internal (None, retriever) tuple.
    """
    return ensure_pubmed_retriever(
        encoder_name=encoder_name,
        top_k=top_k,
        rebuild=False,
        verbose=verbose,
    )[1]


# ---------------------------------------------------------------------
# BM25 implementation
# ---------------------------------------------------------------------

_TOKEN_SPLIT = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Very simple BM25 tokeniser over title+abstract."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_SPLIT.findall(text)]


def _ensure_bm25_index(verbose: bool = True) -> None:
    """
    Build (or reuse) the BM25 index over the PubMed corpus.

    Uses the same PMIDs and text (title+abstract) as the encoder corpora.
    """
    global _BM25_INDEX, _BM25_PMIDS, _BM25_TOKENS

    if _BM25_INDEX is not None:
        return

    # Reuse MiniLM corpus PMIDs / cache
    _ensure_minilm_corpus(verbose=verbose)
    pmids = _ENCODER_CORPORA[_default_minilm_name()]["pmids"]

    if verbose and config.VERBOSE >= 1:
        print(f"[bm25] building BM25 index over {len(pmids)} documents")

    tokens_list: List[List[str]] = []
    for pmid in pmids:
        try:
            rec = _load_cached_pubmed_json(pmid)
            txt = _concat_title_abs(rec)
        except Exception:
            txt = ""
        tokens_list.append(_tokenize(txt))

    _BM25_PMIDS = list(pmids)
    _BM25_TOKENS = tokens_list
    _BM25_INDEX = BM25Okapi(tokens_list)


def retrieve_bm25(
    query: str,
    top_k: int = config.TOP_K_DEFAULT,
) -> List[Any]:
    """
    Real BM25 retrieval.

    Builds a BM25Okapi index over title+abstract and returns the
    top_k papers with highest BM25 scores.
    """
    _ensure_bm25_index(verbose=False)

    assert _BM25_INDEX is not None
    query_tokens = _tokenize(query)
    scores = _BM25_INDEX.get_scores(query_tokens)

    k = min(top_k, len(scores))
    if k <= 0:
        return []

    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    class Doc:
        __slots__ = ("doc_id", "score")

        def __init__(self, pmid: str, score: float):
            self.doc_id = pmid
            self.score = score

    return [Doc(_BM25_PMIDS[i], float(scores[i])) for i in idx]

def retrieve_hybrid_bm25_vector(
    query: str,
    encoder_name: Optional[str] = None,
    top_k: int = config.TOP_K_DEFAULT,
    bm25_k: Optional[int] = None,
    w_bm25: float = 0.7,
    w_sem: float = 0.3,
) -> List[Any]:
    """
    Hybrid retrieval: BM25 candidate generation + dense re-scoring.

    Steps:
      1) Use BM25 to get `bm25_k` lexical candidates.
      2) Compute dense cosine similarities for the same candidates
         using the *same encoder and vectors* as PubMedRetriever.
      3) Combine scores: w_bm25 * bm25_norm + w_sem * cos_norm.
      4) Return top_k docs as Doc objects (doc_id, score).

    Parameters
    ----------
    query:
        Natural language query string.
    encoder_name:
        "minilm", "biobert", or HF id resolvable via _canonical_encoder_name.
        If None, defaults to the MiniLM encoder.
    top_k:
        Number of final documents to return (used for evaluation).
    bm25_k:
        Number of BM25 candidates to generate *before* dense re-ranking.
        If None, uses max(top_k, config.HYBRID_BM25_CANDIDATES).
    """
    # 1) Decide BM25 candidate size
    if bm25_k is None:
        bm25_k = max(
            int(top_k),
            int(getattr(config, "HYBRID_BM25_CANDIDATES", top_k)),
        )

    # 2) BM25 candidate generation
    bm25_docs = retrieve_bm25(query, top_k=bm25_k)
    if not bm25_docs:
        return []

    # 3) Use the SAME encoder & vectors as PubMedRetriever
    enc_alias = _canonical_encoder_name(encoder_name)
    retr = get_retriever_safe(
        encoder_name=enc_alias,
        top_k=max(bm25_k, top_k),
        verbose=False,
    )

    # Query vector in the same space as retr._vecs
    q_vec = retr._qvec(query)             # shape: (d,)
    corpus_vecs = retr._vecs              # shape: (N, d)
    corpus_pmids = retr._pmids

    # Map pmid -> index in this retriever's corpus
    pmid_to_idx: Dict[str, int] = {p: i for i, p in enumerate(corpus_pmids)}

    cand_pmids: List[str] = []
    bm25_scores: List[float] = []
    cand_indices: List[int] = []

    for d in bm25_docs:
        pmid = d.doc_id
        idx = pmid_to_idx.get(pmid)
        if idx is not None:
            cand_pmids.append(pmid)
            bm25_scores.append(float(d.score))
            cand_indices.append(idx)

    if not cand_indices:
        # Fallback: nothing aligns with dense corpus; just return BM25 top_k
        return bm25_docs[:top_k]

    bm25_scores_arr = np.array(bm25_scores, dtype=float)
    cand_vecs = corpus_vecs[cand_indices]   # shape: (n_cand, d)
    cos_scores_arr = cand_vecs @ q_vec     # shape: (n_cand,)

    # Normalisation helpers
    def _min_max_norm(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max <= x_min:
            # All equal; return zeros to avoid NaNs
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    bm25_norm = _min_max_norm(bm25_scores_arr)
    cos_norm = _min_max_norm(cos_scores_arr)

    # 4) Combined score & top_k selection
    combined = w_bm25 * bm25_norm + w_sem * cos_norm
    k = min(int(top_k), len(combined))
    if k <= 0:
        return []

    order = np.argsort(-combined)[:k]

    class Doc:
        __slots__ = ("doc_id", "score")

        def __init__(self, doc_id: str, score: float):
            self.doc_id = doc_id
            self.score = score

    out: List[Doc] = []
    for i in order:
        out.append(Doc(cand_pmids[i], float(combined[i])))

    return out


# ---------------------------------------------------------------------
# MMR re-ranking
# ---------------------------------------------------------------------

def _mmr_rerank(
    pmids: List[str],
    scores: np.ndarray,
    vecs: np.ndarray,
    top_k: int,
    lambda_: float,
) -> List[int]:
    """
    Simple Maximal Marginal Relevance (MMR) re-ranking.
    """
    if top_k <= 0 or len(pmids) == 0:
        return []

    top_k = min(top_k, len(pmids))
    selected: List[int] = []
    remaining = list(range(len(pmids)))

    # Pre-normalised vecs make dot products cosine similarities.
    sim_matrix = vecs @ vecs.T

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick: highest relevance
            best_idx = max(remaining, key=lambda i: scores[i])
        else:
            best_idx = None
            best_score = None
            for i in remaining:
                max_sim = max(sim_matrix[i, j] for j in selected)
                mmr_score = lambda_ * scores[i] - (1.0 - lambda_) * max_sim
                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ---------------------------------------------------------------------
# High-level dense retrieval
# ---------------------------------------------------------------------

def retrieve_vector(
    query: str,
    encoder_name: Optional[str] = None,
    top_k: int = config.TOP_K_DEFAULT,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
) -> List[Any]:
    """
    Vector-based retrieval for RAGulate.

    This function supports multiple encoders (MiniLM, BioBERT) and
    optional MMR re-ranking. It returns a list of lightweight Doc
    objects with ``.doc_id`` and ``.score`` attributes, which is the
    format expected by the pipeline.
    """
    enc_alias = _canonical_encoder_name(encoder_name)
    retr = get_retriever_safe(encoder_name=enc_alias, top_k=top_k, verbose=False)
    hits = retr.retrieve(query, top_k=top_k)

    # No MMR: just wrap hits
    if not use_mmr or len(hits) == 0:
        class Doc:
            __slots__ = ("doc_id", "score")

            def __init__(self, pmid: str, score: float):
                self.doc_id = pmid
                self.score = score

        return [Doc(h["pmid"], h["score"]) for h in hits]

    # With MMR we need per-hit embeddings. We re-use the encoder corpus
    # vectors for the corresponding indices.
    corp = _ENCODER_CORPORA[enc_alias]
    pmid_to_idx = {pmid: i for i, pmid in enumerate(corp["pmids"])}

    aligned_pmids: List[str] = []
    aligned_scores: List[float] = []
    aligned_indices: List[int] = []
    for h in hits:
        pmid = h["pmid"]
        if pmid in pmid_to_idx:
            aligned_pmids.append(pmid)
            aligned_scores.append(float(h["score"]))
            aligned_indices.append(pmid_to_idx[pmid])

    if not aligned_indices:
        # Fallback: no indices matched, just return in original order
        class Doc:
            __slots__ = ("doc_id", "score")

            def __init__(self, pmid: str, score: float):
                self.doc_id = pmid
                self.score = score

        return [Doc(h["pmid"], h["score"]) for h in hits]

    hit_vecs = corp["vecs"][aligned_indices]
    hit_scores = np.array(aligned_scores, dtype=float)
    order = _mmr_rerank(
        pmids=aligned_pmids,
        scores=hit_scores,
        vecs=hit_vecs,
        top_k=min(top_k, len(aligned_pmids)),
        lambda_=mmr_lambda,
    )

    class Doc:
        __slots__ = ("doc_id", "score")

        def __init__(self, pmid: str, score: float):
            self.doc_id = pmid
            self.score = score

    return [Doc(aligned_pmids[i], float(hit_scores[i])) for i in order]


# ---------------------------------------------------------------------
# Vanilla RAG via LlamaIndex
# ---------------------------------------------------------------------

def retrieve_vanilla_rag(
    query: str,
    top_k: int = config.TOP_K_DEFAULT,
) -> List[Any]:
    """
    True 'vanilla RAG' retrieval via LlamaIndex.

    This uses a VectorStoreIndex over the PubMed corpus, built with
    LlamaIndex and a HuggingFace embedding model. It returns a list
    of objects with a ``doc_id`` (pmid) and ``score`` attribute.
    """
    return llamaindex_utils.llamaindex_retrieve(query, top_k=int(top_k))


__all__ = [
    "PubMedRetriever",
    "ensure_pubmed_retriever",
    "get_retriever_safe",
    "retrieve_bm25",
    "retrieve_vector",
    "retrieve_vanilla_rag",
    "retrieve_hybrid_bm25_vector",
]


"""
LlamaIndex-based 'vanilla RAG' utilities for RAGulate.

This module builds a VectorStoreIndex over the PubMed corpus using
LlamaIndex and provides a retrieval function that returns objects
with `.doc_id` and `.score`, compatible with the existing pipeline.
"""

from typing import List, Any, Optional

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from . import config, state
from .embedding import _load_cached_pubmed_json, _concat_title_abs


# Simple module-level cache so we don't rebuild the index repeatedly
_LLAMA_INDEX: Optional[VectorStoreIndex] = None
_LLAMA_EMBED = None


def _build_documents_from_pubmed(max_docs: Optional[int] = None) -> List[Document]:
    """
    Build LlamaIndex Documents from the cached PubMed JSON corpus.

    Each document consists of title+abstract text and a 'pmid' in metadata.
    """
    if not state.GLOBAL_PMIDS:
        # We rely on state.GLOBAL_PMIDS being populated (e.g. by
        # retrieval.ensure_pubmed_retriever / _ensure_global_corpus).
        raise RuntimeError(
            "state.GLOBAL_PMIDS is empty. Make sure the global corpus "
            "has been initialized before building the LlamaIndex index."
        )

    if max_docs is None:
        pmids = state.GLOBAL_PMIDS
    else:
        pmids = state.GLOBAL_PMIDS[:max_docs]

    docs: List[Document] = []

    for pmid in pmids:
        try:
            rec = _load_cached_pubmed_json(pmid)
            txt = _concat_title_abs(rec)
            if not txt:
                continue
            docs.append(Document(text=txt, metadata={"pmid": pmid}))
        except Exception:
            # Skip badly formatted or missing records
            continue

    return docs


def get_llamaindex_index(rebuild: bool = False) -> VectorStoreIndex:
    """
    Build (or reuse) a VectorStoreIndex over the PubMed corpus.

    Uses a HuggingFaceEmbedding model specified by
    config.LLAMAINDEX_EMBED_MODEL_NAME or defaults to
    "sentence-transformers/all-MiniLM-L6-v2".
    """
    global _LLAMA_INDEX, _LLAMA_EMBED

    if _LLAMA_INDEX is not None and not rebuild:
        return _LLAMA_INDEX

    # Choose embedding model
    model_name = getattr(
        config,
        "LLAMAINDEX_EMBED_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    if _LLAMA_EMBED is None:
        _LLAMA_EMBED = HuggingFaceEmbedding(model_name=model_name)

    # Optionally limit max docs for dev (config.LLAMAINDEX_MAX_DOCS)
    max_docs = getattr(config, "LLAMAINDEX_MAX_DOCS", None)
    docs = _build_documents_from_pubmed(max_docs=max_docs)

    _LLAMA_INDEX = VectorStoreIndex.from_documents(docs, embed_model=_LLAMA_EMBED)
    return _LLAMA_INDEX


def llamaindex_retrieve(query: str, top_k: int) -> List[Any]:
    """
    Retrieve documents for a query using the LlamaIndex vector store.

    Returns a list of small objects with `.doc_id` (pmid) and `.score`,
    so they can be used directly by pipeline.run_retrieval_experiment.
    """
    index = get_llamaindex_index(rebuild=False)
    retriever = index.as_retriever(similarity_top_k=int(top_k))

    nodes = retriever.retrieve(query)

    class Doc:
        __slots__ = ("doc_id", "score")

        def __init__(self, doc_id, score):
            self.doc_id = doc_id
            self.score = score

    out: List[Doc] = []
    for n in nodes:
        # Depending on LlamaIndex version, pmid may sit on n.metadata or n.node.metadata
        meta = {}
        if hasattr(n, "metadata") and isinstance(n.metadata, dict):
            meta = n.metadata
        elif hasattr(n, "node") and hasattr(n.node, "metadata"):
            meta = n.node.metadata

        pmid = meta.get("pmid")
        if pmid is None:
            continue

        score = float(getattr(n, "score", 0.0))
        out.append(Doc(pmid, score))

    return out


__all__ = ["get_llamaindex_index", "llamaindex_retrieve"]

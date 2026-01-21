"""
Shared mutable state for the RAGulate benchmark.

The notebook relied on a handful of global variables and
singleton registries to cache models, embeddings and other
artefacts.  This module centralizes those globals so they can be
imported and shared across modules.  Modifying these objects
affects all parts of the pipeline, so tread carefully.
"""

from typing import Any, Dict, List, Optional

# Registry used to track loaded models and artefacts.  Each
# sub-entry holds a ``name`` and the actual ``obj``.  See
# ``llm_models`` and ``retrieval`` for how these fields are
# populated and reused.
_SINGLETONS: Dict[str, Dict[str, Any]] = {
    'sentence_embedder': {'name': None, 'obj': None},
    'biogpt':            {'name': None, 'obj': None},
    'mistral':           {'name': None, 'obj': None},
    'llama31':           {'name': None, 'obj': None},
    'phi3':              {'name': None, 'obj': None},
    'qwen25':            {'name': None, 'obj': None},
    'retriever':         {'key':  None, 'obj': None},
    'global_corpus':     {'built': False, 'num_docs': 0, 'dim': 0},
}

# Global arrays used to hold the PubMed corpus and its vector
# representations.  ``GLOBAL_PMIDS`` is a list of strings.  The
# other arrays are numpy objects initialised in
# ``embedding._ensure_global_corpus``.
GLOBAL_PMIDS: List[str] = []
GLOBAL_TEXTS: List[str] = []  # will be overwritten with np.empty
GLOBAL_VECS: Any = None
GLOBAL_EMB: Any = None

# Query and context caches used by the retriever and RAG logic.
_QUERY_EMB_CACHE: Dict[str, Any] = {}
_RAG_CTX_CACHE: Dict[Any, str] = {}
_SENT_EMB_CACHE: Dict[Any, Any] = {}

__all__ = [
    "_SINGLETONS",
    "GLOBAL_PMIDS",
    "GLOBAL_TEXTS",
    "GLOBAL_VECS",
    "GLOBAL_EMB",
    "_QUERY_EMB_CACHE",
    "_RAG_CTX_CACHE",
    "_SENT_EMB_CACHE",
]
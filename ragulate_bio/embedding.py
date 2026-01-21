"""
Sentence embedding utilities and global corpus handling.

This module provides helpers for constructing and caching a global
corpus of PubMed documents as dense vectors. It mirrors the
behaviour of the original notebook but exposes a clean API for
loading and saving embeddings. The corpus is stored in shared
module state (modules.state) so that multiple calls to the
retriever share the same vectors and avoid recomputation.
"""

import os
import json
from typing import Iterable, List, Optional

import numpy as np
import torch

from . import config, state

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


def get_sentence_embedder(name: Optional[str] = None):
    """Return (and lazily instantiate) a sentence embedding model.

    A singleton instance is cached based on the model name to avoid
    repeatedly loading large embedding models from disk. The default
    model name comes from modules.config.
    """
    model_name = name or config.EMBED_MODEL_NAME
    reg = state._SINGLETONS["sentence_embedder"]
    if reg["obj"] is not None and reg["name"] == model_name:
        return reg["obj"]
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers package is not installed")
    if config.VERBOSE >= 1:
        print(f"[init] Sentence embedder -> {model_name}")
    emb = SentenceTransformer(model_name)
    reg["name"], reg["obj"] = model_name, emb
    if config.VERBOSE >= 1:
        print("[ready] Sentence embedder initialized")
    return emb


def _iter_pubmed_pmids(cache_dir: str) -> Iterable[str]:
    """Yield all PMIDs that have a corresponding JSON file in cache_dir."""
    if not os.path.isdir(cache_dir):
        return []
    for fn in os.listdir(cache_dir):
        if fn.endswith(".json"):
            yield os.path.splitext(fn)[0]


def _load_cached_pubmed_json(pmid: str) -> dict:
    """Load a cached PubMed record from disk."""
    path = os.path.join(config.PUBMED_CACHE, f"{pmid}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _concat_title_abs(rec: dict) -> str:
    """Concatenate the title and abstract of a PubMed record into one string."""
    title = (rec.get("title") or "").strip()
    abstr = (rec.get("abstract") or "").strip()
    return (title + ". " + abstr).strip() if (title or abstr) else ""


def _ensure_global_corpus(embedder_name: Optional[str] = None, verbose: bool = True) -> None:
    """Ensure that the global corpus has been built.

    This populates state.GLOBAL_PMIDS, state.GLOBAL_VECS and related
    globals if they have not already been initialised. It first tries
    to load from a compressed cache on disk; failing that, it falls
    back to reading raw JSON files and encoding them.
    """
    # Avoid re-building if already built
    if state._SINGLETONS["global_corpus"]["built"]:
        if verbose and config.VERBOSE >= 2:
            print("[reuse] global corpus already built")
        return

    embedder = get_sentence_embedder(embedder_name)

    # Try to load from disk cache first
    try:
        if os.path.isfile(config.EMB_CACHE_NPZ) and os.path.isfile(config.EMB_CACHE_META):
            with open(config.EMB_CACHE_META, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("embedder_name") == (embedder_name or config.EMBED_MODEL_NAME):
                npz = np.load(config.EMB_CACHE_NPZ, allow_pickle=True)
                pmids = list(npz["pmids"])
                vecs = np.array(npz["vecs"], np.float32)

                # Populate shared state
                state.GLOBAL_PMIDS[:] = pmids
                state.GLOBAL_TEXTS[:] = []  # not needed after build
                state.GLOBAL_VECS = vecs
                state.GLOBAL_EMB = embedder

                state._SINGLETONS["global_corpus"]["built"] = True
                state._SINGLETONS["global_corpus"]["num_docs"] = len(pmids)
                state._SINGLETONS["global_corpus"]["dim"] = 0 if vecs.size == 0 else vecs.shape[1]

                if verbose and config.VERBOSE >= 1:
                    print(f"[global-corpus] loaded cache: {len(pmids)} docs, dim={state.GLOBAL_VECS.shape[1]}")
                return
    except Exception as e:
        print(f"[warn] failed to load embedding cache, rebuilding: {e}")

    # Otherwise rebuild from raw JSON
    pmids: List[str] = []
    texts: List[str] = []
    for pmid in _iter_pubmed_pmids(config.PUBMED_CACHE):
        try:
            rec = _load_cached_pubmed_json(pmid)
            txt = _concat_title_abs(rec)
            if txt:
                pmids.append(pmid)
                texts.append(txt)
        except Exception as e:
            if verbose:
                print(f"[warn] skipping pmid {pmid}: {e}")

    # Encode texts into vectors
    use_cuda = hasattr(embedder, "encode") and bool(getattr(embedder, "device", None))
    batch_sz = 512 if use_cuda else 128
    if texts:
        with torch.inference_mode():
            vecs = embedder.encode(
                texts,
                batch_size=batch_sz,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                device=("cuda" if use_cuda else "cpu"),
            )
    else:
        vecs = np.zeros((0, 384), np.float32)

    # Populate shared state
    state.GLOBAL_PMIDS[:] = pmids
    state.GLOBAL_TEXTS[:] = texts
    state.GLOBAL_VECS = vecs.astype(np.float32, copy=False)
    state.GLOBAL_EMB = embedder

    state._SINGLETONS["global_corpus"]["built"] = True
    state._SINGLETONS["global_corpus"]["num_docs"] = len(pmids)
    state._SINGLETONS["global_corpus"]["dim"] = 0 if vecs.size == 0 else vecs.shape[1]

    # Save cache for subsequent runs
    try:
        os.makedirs(config.PUBMED_CACHE, exist_ok=True)
        np.savez_compressed(
            config.EMB_CACHE_NPZ,
            pmids=np.array(pmids, dtype=object),
            vecs=state.GLOBAL_VECS,
        )
        with open(config.EMB_CACHE_META, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedder_name": (embedder_name or config.EMBED_MODEL_NAME),
                    "num_docs": len(pmids),
                    "dim": state._SINGLETONS["global_corpus"]["dim"],
                },
                f,
            )
        if verbose and config.VERBOSE >= 1:
            print(f"[global-corpus] built & cached: {len(pmids)} docs, dim={state._SINGLETONS['global_corpus']['dim']}")
    except Exception as e:
        print(f"[warn] could not save embedding cache: {e}")


__all__ = [
    "get_sentence_embedder",
    "_ensure_global_corpus",
    "_iter_pubmed_pmids",
    "_load_cached_pubmed_json",
    "_concat_title_abs",
]

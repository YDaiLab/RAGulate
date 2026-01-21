"""
Configuration constants for the RAGulate benchmark.

These constants centralize all of the tunable knobs used by the
benchmark.  When adapting the pipeline to new corpora or models
you should modify values here rather than scattering magic numbers
throughout the codebase.  Most values mirror those found in the
original notebook.  No secrets (such as API tokens) are stored
here; authentication tokens should be provided via environment
variables at runtime.
"""

import os
from pathlib import Path

#: Random seed used for numpy and random for reproducibility.
SEED: int = 42


#: LLM debugging / logging
#: When True, _llm_generate_batch will print prompts and outputs for
#: a few examples to stdout. Turn this off for large runs.
LLM_DEBUG: bool = False
# Max number of examples to log per batch when LLM_DEBUG is True
LLM_DEBUG_MAX_EXAMPLES: int = 5
# Optional: truncate long prompts in debug output
LLM_DEBUG_MAX_PROMPT_CHARS: int = 400


# ---------------------------------------------------------------------------
# Combination hyperparameters for RAGulate scoring
# ---------------------------------------------------------------------------
# When combining the LLM-derived probability (L) and the retrieval support
# score (S), the pipeline uses a mixture of a linear term and a multiplicative
# (geometric) term.  The linear term is:
#     L_linear = lambda_linear * L + (1 - lambda_linear) * S
# and the multiplicative term is:
#     L_mult   = (L + eps)**alpha_mult * (S + eps)**(1 - alpha_mult)
# The final combined score is:
#     final_score = (1 - beta) * L_linear + beta * L_mult
# All hyperparameters live in [0,1].  Set beta=0.0 to disable the
# multiplicative component.

COMBINE_LAMBDA_LINEAR: float = 0.5  # weight on L in the linear term
COMBINE_ALPHA_MULT: float = 0.5     # exponent weight on L in the multiplicative term
COMBINE_BETA: float = 0.0           # weight on multiplicative vs linear term
COMBINE_EPS: float = 1e-8           # small constant to avoid zero when computing L_mult


#: Directory where persistent artefacts (such as embedding
#: caches) will be stored.  These paths are relative to the
#: working directory when the scripts are executed.
PERSIST_DIR: str = "./persist"

#: Directory where PubMed JSON and embedding caches live.  The
#: retriever will look for ``*.json`` files here and will write
#: out compressed embedding matrices for faster startup.
PUBMED_CACHE: str = "./pubmed_cache"

#: Default number of documents to retrieve in the vector search.
TOP_K_DEFAULT: int = 8

# How many documents BM25 should retrieve before re-ranking in the hybrid retriever
HYBRID_BM25_CANDIDATES = 300  # try 100, 200, 300

#: Primary ``k`` used when retrieving documents for each TF–gene
#: query during benchmarking.  This is separated from
#: ``TOP_K_DEFAULT`` so you can experiment with different k
#: values without impacting other parts of the code.
TOP_K_RETRIEVE: int = 50

#: Name of the sentence embedding model used for PubMed passage
#: encoding.  This is the default model; other models can be
#: passed at runtime through the configuration API.
EMBED_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
# BioBERT sentence-embedding model for retrieval (R2/R4/R5)
BIOBERT_SENTENCE_MODEL_NAME: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"


#: BioBERT (or Bio-domain) embedding model for an alternative corpus.
#: This is used for encoder_name="biobert" in retrieval.py.
#: If you prefer another HF sentence-transformers model, change it here.
BIOBERT_EMBED_MODEL_NAME: str = BIOBERT_SENTENCE_MODEL_NAME

#: Embedding model configuration for LlamaIndex-based vanilla RAG.
LLAMAINDEX_EMBED_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
LLAMAINDEX_MAX_DOCS = 10000  # or an int like 50000 for dev

#: Names of the causal language models used for question
#: answering.  These strings correspond to HuggingFace model ids.
BIOGPT_MODEL_NAME: str = "microsoft/BioGPT-Large"
MISTRAL_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
LLAMA31_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3_MODEL_NAME: str = "microsoft/Phi-3-mini-128k-instruct"
QWEN25_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
OPENAI_MODEL_NAME: str = "openai/gpt-oss-20b"

#: Flags controlling which models participate in the benchmark.
#: Set these to ``False`` to skip a model entirely.
USE_MISTRAL: bool = True
USE_LLAMA31: bool = True
USE_PHI3: bool = True
USE_QWEN25: bool = True
USE_OPENAI: bool = False

#: Generation parameters for the LLMs.  ``GEN_MAX_NEW`` controls
#: how many new tokens to generate per call.  ``SAFETY_MARGIN``
#: reserves some context tokens for metadata and RAG context.
GEN_MAX_NEW: int = 8
SAFETY_MARGIN: int = 64

#: Controls how many sentences are selected from each paper when
#: constructing the RAG context.  ``MAX_PMIDS_PER_EDGE`` caps
#: the number of papers considered per gold-standard edge.
MAX_SENT_PER_PAPER: int = 3
MAX_PMIDS_PER_EDGE: int = 25

#: Number of epochs to run when calling ``run_benchmark_epochs``.
EPOCHS: int = 5

#: How often to log progress within an epoch (not currently used).
LOG_EVERY: int = 1

#: Whether to perform permutation testing during evaluation.
DO_PERMUTE: bool = False

#: Number of permutations used in the permutation test.  Larger
#: values give tighter confidence intervals but take longer.
N_PERM: int = 100

#: Derived cache filenames for sentence embeddings.  These are
#: built from the embedding model name and the ``PUBMED_CACHE``
#: directory.  Changing the embedder will automatically point to
#: a different cache file.
EMB_CACHE_BASENAME: str = f"embeddings-{(EMBED_MODEL_NAME or '').replace('/', '__')}"
EMB_CACHE_NPZ: str = os.path.join(PUBMED_CACHE, EMB_CACHE_BASENAME + ".npz")
EMB_CACHE_META: str = os.path.join(PUBMED_CACHE, EMB_CACHE_BASENAME + ".meta.json")

#: Base output directory for benchmark artefacts (gold standard,
# Project root:  .../RAGulate/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
#: CollectRI pickles, metrics, etc.).
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"


# Data directory: .../RAGulate/data/
DATA_DIR: Path = PROJECT_ROOT / "data"

# Structured resources directory: .../RAGulate/data/structured/
STRUCTURED_DATA_DIR: Path = DATA_DIR / "structured"

# HGNC complete set (TSV) from genenames.org
HGNC_COMPLETE_SET_FILE: Path = STRUCTURED_DATA_DIR / "hgnc_complete_set.txt"

#: Path to preprocessed CollectRI documents with context information.
#: This should point to the pickle file you generated earlier.
COLLECTRI_DOCS_PATH: Path = OUTPUT_DIR / "collectri_docs.pkl"

#: Default input files and output artefact paths.  You can
#: override these when calling the IO functions if your data
#: lives elsewhere.

# Gold-standard CSV for the RAGulate benchmark (v2).
GOLD_PATH: Path = OUTPUT_DIR / "ragulate_gold_standard_v2.csv"

# Benchmark outputs for multi-model retrieval / classification runs.
ROWS_CSV: Path = OUTPUT_DIR / "retrieval_benchmark_all_models.csv"
METRICS_JSON: Path = OUTPUT_DIR / "retrieval_benchmark_all_models_metrics.json"

# Max new tokens for RAGulate HTML / CSV summaries
LLM_SUMMARY_MAX_NEW = 160

#: Verbosity level for internal logging.  0 = quiet, 1 =
#: high-level messages, 2 = detailed messages.  This flag is
#: referenced by various modules to decide how much to print.
VERBOSE: int = 1

__all__ = [
    "SEED",
    "PERSIST_DIR",
    "PUBMED_CACHE",
    "TOP_K_DEFAULT",
    "HYBRID_BM25_CANDIDATES",
    "TOP_K_RETRIEVE",
    "EMBED_MODEL_NAME",
    "BIOBERT_SENTENCE_MODEL_NAME",
    "BIOBERT_EMBED_MODEL_NAME",
    "LLAMAINDEX_EMBED_MODEL_NAME",
    "LLAMAINDEX_MAX_DOCS",
    "BIOGPT_MODEL_NAME",
    "MISTRAL_MODEL_NAME",
    "LLAMA31_MODEL_NAME",
    "PHI3_MODEL_NAME",
    "QWEN25_MODEL_NAME",
    "OPENAI_MODEL_NAME",
    "USE_MISTRAL",
    "USE_LLAMA31",
    "USE_PHI3",
    "USE_QWEN25",
    "USE_OPENAI",
    "GEN_MAX_NEW",
    "SAFETY_MARGIN",
    "MAX_SENT_PER_PAPER",
    "MAX_PMIDS_PER_EDGE",
    "EPOCHS",
    "LOG_EVERY",
    "DO_PERMUTE",
    "N_PERM",
    "EMB_CACHE_BASENAME",
    "EMB_CACHE_NPZ",
    "EMB_CACHE_META",
    "OUTPUT_DIR",
    "COLLECTRI_DOCS_PATH",
    "GOLD_PATH",
    "ROWS_CSV",
    "METRICS_JSON",
    "VERBOSE",
    "PROJECT_ROOT",
    "DATA_DIR",
    "STRUCTURED_DATA_DIR",
    "HGNC_COMPLETE_SET_FILE",
    "COMBINE_LAMBDA_LINEAR",
    "COMBINE_ALPHA_MULT",
    "COMBINE_BETA",
    "COMBINE_EPS",
]

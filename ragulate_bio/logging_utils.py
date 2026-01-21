"""
Helper functions for printing configuration and singleton status.

These functions are useful when running benchmarks interactively
to understand which models and components have been loaded.
"""

from .state import _SINGLETONS
from . import config

def print_singleton_status() -> None:
    """Print the status of cached singletons to stdout.

    This function reports which sentence encoder and language models
    have been initialised and whether the global corpus has been
    built.  It mirrors the diagnostic output from the original
    notebook.
    """
    s = _SINGLETONS
    print('[status] sentence_embedder:', bool(s['sentence_embedder']['name']))
    print('[status] biogpt:', s['biogpt']['name'])
    print('[status] mistral:', s['mistral']['name'])
    print('[status] llama31:', s['llama31']['name'])
    print('[status] phi3:', s['phi3']['name'])
    print('[status] qwen2.5:', s['qwen25']['name'])
    print('[status] retriever:', s['retriever']['key'])
    gc = s['global_corpus']
    print(f"[status] corpus: built={gc['built']} n_docs={gc['num_docs']} dim={gc['dim']}")


def print_config_banner(n_rows: int, epochs: int) -> None:
    """Pretty-print a summary of the current benchmark configuration.

    Parameters
    ----------
    n_rows:
        Number of rows in the gold dataset that will be used for the
        benchmark.
    epochs:
        Number of epochs to run when using the epoch-based benchmark
        helper.
    """
    enabled = [name for name, flag in [
        ('Mistral', config.USE_MISTRAL),
        ('Llama3.1', config.USE_LLAMA31),
        ('Phi-3', config.USE_PHI3),
        ('Qwen2.5', config.USE_QWEN25),
    ] if flag]

    gc = _SINGLETONS['global_corpus']
    corpus_str = f"built={gc['built']}  n_docs={gc['num_docs']}  dim={gc['dim']}"

    print("\n===== RAGulate Benchmark Config =====")
    print(f"Rows: {n_rows}")
    print(f"Epochs: {epochs}")
    print(f"Retriever top-k: {config.TOP_K_RETRIEVE}")
    print(f"Corpus: {corpus_str}")
    print(f"Models: BioGPT{'' if not enabled else ', ' + ', '.join(enabled)}")
    print(f"Generation: max_new={config.GEN_MAX_NEW}  safety_margin={config.SAFETY_MARGIN}")
    print(f"RAG: max_sent_per_paper={config.MAX_SENT_PER_PAPER}  max_pmids_per_edge={config.MAX_PMIDS_PER_EDGE}")
    print(f"Permutations: {config.N_PERM if config.DO_PERMUTE else 0}  (enabled={config.DO_PERMUTE})")
    print("====================================\n")

__all__ = ["print_singleton_status", "print_config_banner"]
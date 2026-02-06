# ragulate_bio/__init__.py

__version__ = "1.1.2"

def about():
    return {
        "name": "Ragulate-Bio",
        "version": __version__,
        "summary": "Retrieval augmented validation for TF inference in single cell biology",
    }

# Expose public submodules
from . import (
    config,
    state,
    logging_utils,
    embedding,
    retrieval,
    llm_models,
    rag,
    metrics,
    pipeline,
    io_utils,
    plots,
)

__all__ = [
    "about",
    "__version__",
    "config",
    "state",
    "logging_utils",
    "embedding",
    "retrieval",
    "llm_models",
    "rag",
    "metrics",
    "pipeline",
    "io_utils",
    "plots",
]

# Optional gentle conflict warning so users know about the other 'ragulate'
def _warn_if_conflicting():
    try:
        import importlib.metadata as im

        names = {d.metadata["Name"].lower() for d in im.distributions()}
        if "ragulate" in names:
            import warnings

            warnings.warn(
                "Another package named 'ragulate' is installed. "
                "Use 'import ragulate_bio' for this package."
            )
    except Exception:
        pass


_warn_if_conflicting()
del _warn_if_conflicting

"""
Helpers for evaluating retrieval with and without HGNC alias expansion.

This module centralises the notebook logic for:
  * running alias vs no-alias retrieval
  * computing per-edge Recall@K / Precision@K distributions
  * bootstrapping mean ± CI
  * plotting publication-style bar plots

The goal is to keep notebooks clean while preserving explicit control
over each step (run → summarise → plot).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from .pipeline import (
    run_retrieval_experiment,
    compute_substring_upper_bound,
    compute_substring_upper_bound_precision,
)
from . import metrics as metric_utils


# -------------------------------------------------------------------
# Global colours (lighter, more vibrant, still publication-friendly)
# -------------------------------------------------------------------

# Default palette – can be overridden in plotting functions
COLOR_NO_ALIAS = "#5DA5DA"  # bright blue
COLOR_ALIAS = "#FAA43A"     # warm orange


# -------------------------------------------------------------------
# 1. Retrieval runner
# -------------------------------------------------------------------

def run_alias_vs_noalias(
    gold_df,
    *,
    method_type: str = "hybrid",
    encoder_name: Optional[str] = "minilm",
    top_k: int = 50,
    positives_only: bool = True,
    query_mode: str = "tf_tg_ctx",
    expand_context: bool = False,
    max_ctx_terms: int = 4,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
    alias_max: Optional[int] = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run retrieval twice on the same gold_df:
      (1) without HGNC alias expansion
      (2) with HGNC alias expansion

    Returns
    -------
    rows_no_alias, rows_alias : list of dict
        Raw retrieval rows as returned by ``run_retrieval_experiment``.
    """
    rows_no_alias = run_retrieval_experiment(
        gold_df,
        method_type=method_type,
        encoder_name=encoder_name,
        top_k=top_k,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        positives_only=positives_only,
        query_mode=query_mode,
        expand_context=expand_context,
        max_ctx_terms=max_ctx_terms,
        use_aliases=False,
        alias_max=alias_max,
    )

    rows_alias = run_retrieval_experiment(
        gold_df,
        method_type=method_type,
        encoder_name=encoder_name,
        top_k=top_k,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        positives_only=positives_only,
        query_mode=query_mode,
        expand_context=expand_context,
        max_ctx_terms=max_ctx_terms,
        use_aliases=True,
        alias_max=alias_max,
    )

    return rows_no_alias, rows_alias


# -------------------------------------------------------------------
# 2. Metric collection + bootstrapping
# -------------------------------------------------------------------

def _collect_edge_metric(
    rows: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    metric_fn,
) -> Dict[int, List[float]]:
    """
    Collect per-edge metric@K over all edges.

    Parameters
    ----------
    rows :
        Output rows from ``run_retrieval_experiment``.
    ks :
        List of K values (e.g. [1, 3, 5, 10, 20, 50, 100]).
    metric_fn :
        Function of the form metric_fn(gold_pmids, retrieved_pmids, k)
        such as ``metrics.recall_at_k`` or ``metrics.precision_at_k``.
    """
    per_k: Dict[int, List[float]] = {int(k): [] for k in ks}
    for r in rows:
        gold = r["gold_pmids"]
        ret = r["retrieved_pmids"]
        for k in ks:
            per_k[int(k)].append(float(metric_fn(gold, ret, int(k))))
    return per_k


def _bootstrap_mean_ci(
    per_k: Dict[int, Sequence[float]],
    ks: Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap mean and (1 - alpha) CI for each K.

    Returns
    -------
    means, lows, highs : np.ndarray
        Arrays (len(ks),) with point estimate and CI bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    means: List[float] = []
    lows: List[float] = []
    highs: List[float] = []

    for k in ks:
        vals = np.asarray(per_k[int(k)], dtype=float)
        if vals.size == 0:
            means.append(0.0)
            lows.append(0.0)
            highs.append(0.0)
            continue

        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(vals, size=vals.shape[0], replace=True)
            boot_means.append(float(sample.mean()))
        boot_means = np.asarray(boot_means, dtype=float)

        mean = float(vals.mean())
        lo, hi = np.percentile(
            boot_means,
            [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)],
        )
        means.append(mean)
        lows.append(float(lo))
        highs.append(float(hi))

    return np.asarray(means), np.asarray(lows), np.asarray(highs)


def _compute_ci_for_metric(
    rows_no_alias: Sequence[Dict[str, Any]],
    rows_alias: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    metric: str,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Compute mean ± CI for a chosen metric ("recall" or "precision")
    for both (no-alias, alias) settings.

    Returns a dict with keys:
      - "ks"
      - "mean_no", "lo_no", "hi_no"
      - "mean_al", "lo_al", "hi_al"
    """
    if metric == "recall":
        metric_fn = metric_utils.recall_at_k
    elif metric == "precision":
        metric_fn = metric_utils.precision_at_k
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    per_no = _collect_edge_metric(rows_no_alias, ks, metric_fn)
    per_al = _collect_edge_metric(rows_alias, ks, metric_fn)

    m_no, lo_no, hi_no = _bootstrap_mean_ci(per_no, ks, n_boot=n_boot, alpha=alpha, rng=rng)
    m_al, lo_al, hi_al = _bootstrap_mean_ci(per_al, ks, n_boot=n_boot, alpha=alpha, rng=rng)

    out: Dict[str, Any] = {
        "ks": np.asarray(ks, dtype=int),
        "mean_no": m_no,
        "lo_no": lo_no,
        "hi_no": hi_no,
        "mean_al": m_al,
        "lo_al": lo_al,
        "hi_al": hi_al,
    }
    return out


# -------------------------------------------------------------------
# 3. Plotting helpers (publication style)
# -------------------------------------------------------------------

def _apply_pub_style() -> None:
    """Set a consistent matplotlib style for the bar plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.facecolor": "#F9FAFB",
        "figure.facecolor": "white",
    })


def plot_recall_alias_ci(
    gold_df,
    rows_no_alias: Sequence[Dict[str, Any]],
    rows_alias: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    outfile: Optional[str] = "retrieval_recall_alias_CI_pub_v2.png",
    use_custom_colors: bool = True,
    color_no_alias: Optional[str] = None,
    color_alias: Optional[str] = None,
) -> None:
    """
    Plot Recall@K comparison (no aliases vs HGNC aliases) with
    bootstrap 95% CIs and substring upper-bound line (recall-based).

    Parameters
    ----------
    use_custom_colors :
        If False, do not set any colour on bars and let Matplotlib's
        default colour cycle handle them.
    color_no_alias, color_alias :
        Optional overrides for bar colours. Only used if
        use_custom_colors=True. If None, fall back to module-level
        COLOR_NO_ALIAS / COLOR_ALIAS.
    """
    ci = _compute_ci_for_metric(
        rows_no_alias,
        rows_alias,
        ks,
        metric="recall",
        n_boot=n_boot,
        alpha=alpha,
        rng=rng,
    )

    # Convert to %
    mean_no_pct = ci["mean_no"] * 100.0
    mean_al_pct = ci["mean_al"] * 100.0

    lo_no_pct = ci["lo_no"] * 100.0
    hi_no_pct = ci["hi_no"] * 100.0
    lo_al_pct = ci["lo_al"] * 100.0
    hi_al_pct = ci["hi_al"] * 100.0

    yerr_no = np.vstack([mean_no_pct - lo_no_pct, hi_no_pct - mean_no_pct])
    yerr_al = np.vstack([mean_al_pct - lo_al_pct, hi_al_pct - mean_al_pct])

    # substring upper bound (recall-based)
    ub = compute_substring_upper_bound(gold_df)
    ub_pct = float(ub["upper_bound_recall"]) * 100.0

    _apply_pub_style()

    fig, ax = plt.subplots(figsize=(6, 3.0))

    x = np.arange(len(ks))
    bar_width = 0.35

    # Build kwargs so we can optionally omit color
    kwargs_no = dict(
        width=bar_width,
        edgecolor="black",
        linewidth=0.6,
        label="No aliases",
    )
    kwargs_al = dict(
        width=bar_width,
        edgecolor="black",
        linewidth=0.6,
        label="HGNC aliases",
    )

    if use_custom_colors:
        c_no = color_no_alias or COLOR_NO_ALIAS
        c_al = color_alias or COLOR_ALIAS
        if c_no is not None:
            kwargs_no["color"] = c_no
        if c_al is not None:
            kwargs_al["color"] = c_al

    bars_no = ax.bar(
        x - bar_width / 2.0,
        mean_no_pct,
        **kwargs_no,
    )
    ax.errorbar(
        x - bar_width / 2.0,
        mean_no_pct,
        yerr=yerr_no,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
    )

    bars_al = ax.bar(
        x + bar_width / 2.0,
        mean_al_pct,
        **kwargs_al,
    )
    ax.errorbar(
        x + bar_width / 2.0,
        mean_al_pct,
        yerr=yerr_al,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
    )

    ax.axhline(
        ub_pct,
        linestyle="--",
        linewidth=1.1,
        color="black",
        label=f"Substring upper bound ({ub_pct:.1f}%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (%)")

    y_max = max(float(mean_no_pct.max()), float(mean_al_pct.max()), ub_pct)
    ax.set_ylim(0.0, y_max * 1.25)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # label alias bars above whiskers
    for rect, mean_val, yerr_hi in zip(bars_al, mean_al_pct, yerr_al[1]):
        y_text = float(mean_val) + float(yerr_hi) + y_max * 0.02
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            y_text,
            f"{mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def plot_precision_alias_ci(
    gold_df,
    rows_no_alias: Sequence[Dict[str, Any]],
    rows_alias: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    outfile: Optional[str] = "retrieval_precision_alias_CI_with_substring_pub.png",
    use_custom_colors: bool = True,
    color_no_alias: Optional[str] = None,
    color_alias: Optional[str] = None,
) -> None:
    """
    Plot Precision@K comparison (no aliases vs HGNC aliases) with
    bootstrap 95% CIs. No substring upper bound is shown here,
    because the substring check only defines a recall upper bound.

    Parameters
    ----------
    use_custom_colors :
        If False, do not set any colour on bars and let Matplotlib's
        default colour cycle handle them.
    color_no_alias, color_alias :
        Optional overrides for bar colours. Only used if
        use_custom_colors=True. If None, fall back to module-level
        COLOR_NO_ALIAS / COLOR_ALIAS.
    """
    ci = _compute_ci_for_metric(
        rows_no_alias,
        rows_alias,
        ks,
        metric="precision",
        n_boot=n_boot,
        alpha=alpha,
        rng=rng,
    )

    # Convert to %
    mean_no_pct = ci["mean_no"] * 100.0
    mean_al_pct = ci["mean_al"] * 100.0

    lo_no_pct = ci["lo_no"] * 100.0
    hi_no_pct = ci["hi_no"] * 100.0
    lo_al_pct = ci["lo_al"] * 100.0
    hi_al_pct = ci["hi_al"] * 100.0

    yerr_no = np.vstack([mean_no_pct - lo_no_pct, hi_no_pct - mean_no_pct])
    yerr_al = np.vstack([mean_al_pct - lo_al_pct, hi_al_pct - mean_al_pct])

    _apply_pub_style()

    fig, ax = plt.subplots(figsize=(5, 3.0))

    x = np.arange(len(ks))
    bar_width = 0.35

    kwargs_no = dict(
        width=bar_width,
        edgecolor="black",
        linewidth=0.6,
        label="No aliases",
    )
    kwargs_al = dict(
        width=bar_width,
        edgecolor="black",
        linewidth=0.6,
        label="HGNC aliases",
    )

    if use_custom_colors:
        c_no = color_no_alias or COLOR_NO_ALIAS
        c_al = color_alias or COLOR_ALIAS
        if c_no is not None:
            kwargs_no["color"] = c_no
        if c_al is not None:
            kwargs_al["color"] = c_al

    bars_no = ax.bar(
        x - bar_width / 2.0,
        mean_no_pct,
        **kwargs_no,
    )
    ax.errorbar(
        x - bar_width / 2.0,
        mean_no_pct,
        yerr=yerr_no,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
    )

    bars_al = ax.bar(
        x + bar_width / 2.0,
        mean_al_pct,
        **kwargs_al,
    )
    ax.errorbar(
        x + bar_width / 2.0,
        mean_al_pct,
        yerr=yerr_al,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K (%)")

    y_max = float(max(mean_no_pct.max(), mean_al_pct.max()))
    ax.set_ylim(0.0, y_max * 1.25)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # label alias bars above whiskers
    for rect, mean_val, yerr_hi in zip(bars_al, mean_al_pct, yerr_al[1]):
        y_text = float(mean_val) + float(yerr_hi) + y_max * 0.02
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            y_text,
            f"{mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()

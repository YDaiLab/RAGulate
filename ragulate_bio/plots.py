"""
Plotting helpers for visualising benchmark results.

These functions produce bar charts and line plots from the metrics
dictionary returned by the benchmark.  The colours and grouping
mirror the original notebook's aesthetic.  Matplotlib is used
exclusively; if it is not available these functions will raise.
"""

from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt  # type: ignore


def _is_rag_key(k: str) -> bool:
    """Return True if the method name indicates a RAG configuration."""
    import re
    return bool(re.search(r'(rag|\+rag|\(rag\))', k, flags=re.I))


def _root_name(k: str) -> str:
    """Strip common RAG markers and mode suffixes from a method name."""
    import re
    k2 = re.sub(r'(\s*\+?\s*rag|\s*\(rag\))', '', k, flags=re.I)
    k2 = re.sub(r'(_?no[-_ ]?rag|_?only)$', '', k2, flags=re.I)
    k2 = re.sub(r'[_\s]+', ' ', k2).strip()
    return k2


def _lighten(color: Tuple[float, float, float], factor: float = 0.55) -> Tuple[float, float, float]:
    """Lighten an RGB tuple by blending towards white."""
    r, g, b = color
    return (1 - (1 - r) * factor, 1 - (1 - g) * factor, 1 - (1 - b) * factor)


def plot_llm_pairs_bar(metrics_dict: Dict[str, Dict[str, Any]], metric: str = 'auroc', title: Optional[str] = None, ylim: Tuple[float, float] = (0, 1)) -> None:
    """Plot a grouped bar chart comparing base vs +RAG for each model."""
    # Group keys into root -> {base, rag}
    groups: Dict[str, Dict[str, Any]] = {}
    for name, vals in metrics_dict.items():
        root = _root_name(name)
        is_rag = _is_rag_key(name)
        d = groups.setdefault(root, {'base': None, 'rag': None, 'names': {'base': None, 'rag': None}})
        val = vals.get(metric)
        if val is None:
            continue
        if is_rag:
            d['rag'] = val
            d['names']['rag'] = name
        else:
            d['base'] = val
            d['names']['base'] = name
    groups = {k: v for k, v in groups.items() if (v['base'] is not None) or (v['rag'] is not None)}
    if not groups:
        print(f'[warn] No values found for metric="{metric}".')
        return
    roots = sorted(groups.keys())
    base_vals = [groups[r]['base'] for r in roots]
    rag_vals = [groups[r]['rag'] for r in roots]
    cmap = plt.get_cmap('tab10')
    base_colors = [cmap(i % 10)[:3] for i in range(len(roots))]
    rag_colors = [_lighten(c, 0.55) for c in base_colors]
    x = range(len(roots))
    width = 0.38
    plt.figure(figsize=(max(6, len(roots) * 1.2), 4.5))
    if title is None:
        title = f'Benchmark — {metric.upper()}'
    plt.title(title)
    base_heights = [v if v is not None else 0 for v in base_vals]
    rag_heights = [v if v is not None else 0 for v in rag_vals]
    b0 = plt.bar([xi - width / 2 for xi in x], base_heights, width, color=base_colors, label='Base (No-RAG)', edgecolor='black', linewidth=0.6)
    b1 = plt.bar([xi + width / 2 for xi in x], rag_heights, width, color=rag_colors, label='+RAG', edgecolor='black', linewidth=0.6)
    def _annotate(bar_container, values: List[Optional[float]]) -> None:
        for rect, val in zip(bar_container, values):
            if val is None:
                continue
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, h + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    _annotate(b0, base_vals)
    _annotate(b1, rag_vals)
    plt.xticks(list(x), roots, rotation=15, ha='right')
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(metric.upper())
    plt.legend(loc='best', frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_metrics_bars(metrics: Dict[str, Dict[str, Any]], title: str = 'Benchmark Summary') -> None:
    """Plot simple bar charts of AUROC and AUPRC for all methods."""
    methods = list(metrics.keys())
    aurocs = [metrics[m].get('auroc') for m in methods]
    auprcs = [metrics[m].get('auprc') for m in methods]
    plt.figure()
    plt.title(f"{title} — AUROC")
    plt.bar(methods, [0 if v is None else v for v in aurocs])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('AUROC')
    plt.show()
    plt.figure()
    plt.title(f"{title} — AUPRC")
    plt.bar(methods, [0 if v is None else v for v in auprcs])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('AUPRC')
    plt.show()


def plot_history(history: List[Dict[str, Any]], metric: str = 'auroc', title: str = 'Epoch Progress') -> None:
    """Plot the progression of a metric across epochs for each method."""
    if not history:
        print('[warn] empty history')
        return
    methods = list(history[0]['metrics'].keys())
    xs = [h['epoch'] for h in history]
    for m in methods:
        ys = [h['metrics'][m].get(metric) if m in h['metrics'] else None for h in history]
        plt.figure()
        plt.title(f"{title} — {m} ({metric.upper()})")
        plt.plot(xs, [0 if v is None else v for v in ys], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.show()


__all__ = ["plot_llm_pairs_bar", "plot_metrics_bars", "plot_history"]
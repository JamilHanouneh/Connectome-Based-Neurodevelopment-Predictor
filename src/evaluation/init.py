"""
Evaluation and metrics
"""

from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_all_metrics,
    print_metrics
)

from .visualization import (
    plot_roc_curves,
    plot_confusion_matrices,
    plot_scatter_plots,
    plot_bland_altman,
    generate_html_report
)

__all__ = [
    'compute_classification_metrics',
    'compute_regression_metrics',
    'compute_all_metrics',
    'print_metrics',
    'plot_roc_curves',
    'plot_confusion_matrices',
    'plot_scatter_plots',
    'plot_bland_altman',
    'generate_html_report'
]

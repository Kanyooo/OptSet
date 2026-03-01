from __future__ import annotations

DEFAULT_DIMS_MVP = [2, 20, 50]
DEFAULT_TARGETS = [
    "quadratic",
    "norm_cone",
    "quadratic_plus_norm",
    "huber",
    "logsumexp",
    "structured_composition",
]
DEFAULT_MODELS = ["relu_icnn", "softplus_icnn", "quad_icnn", "norm_icnn", "soc_icnn"]
DEFAULT_SEEDS = [0]
DEFAULT_BUDGETS = [1000, 3000, 10000, 30000]
DEFAULT_THRESHOLDS = [1e-1, 1e-2, 1e-3]

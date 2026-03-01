from __future__ import annotations

from targets.huber import HuberTarget
from targets.logsumexp import LogSumExpTarget
from targets.norm_cone import NormConeTarget
from targets.quad_norm import QuadNormTarget
from targets.quadratic import QuadraticTarget
from targets.structured_composition import StructuredCompositionTarget


TARGET_REGISTRY = {
    "quadratic": QuadraticTarget,
    "norm_cone": NormConeTarget,
    "quadratic_plus_norm": QuadNormTarget,
    "huber": HuberTarget,
    "logsumexp": LogSumExpTarget,
    "structured_composition": StructuredCompositionTarget,
}


def build_target(name: str, d: int, seed: int):
    return TARGET_REGISTRY[name](d=d, seed=seed)

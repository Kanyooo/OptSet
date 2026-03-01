from __future__ import annotations

from pathlib import Path

from models.elu_icnn import ELU_SKIP_REASON
from models.norm_icnn import NormICNN
from models.quad_icnn import QuadICNN
from models.relu_icnn import ReLUICNN
from models.soc_icnn import SOCICNN
from models.softplus_icnn import SoftplusICNN


def model_builders(include_elu: bool = False):
    builders = {
        "relu_icnn": ReLUICNN,
        "softplus_icnn": SoftplusICNN,
        "quad_icnn": QuadICNN,
        "norm_icnn": NormICNN,
        "soc_icnn": SOCICNN,
    }
    if include_elu:
        print(ELU_SKIP_REASON)
    return builders


def out_paths(root: str):
    p = Path(root)
    return {
        "raw": p / "raw",
        "processed": p / "processed",
        "figures": p / "figures",
        "tables": p / "tables",
    }

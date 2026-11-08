"""Model registry for TEP fault-diagnosis experiments."""

from .baselines import MODEL_CLASSES, build_model
from .qc_hgt import QCHGT

__all__ = ["MODEL_CLASSES", "QCHGT", "build_model"]

# Design note (paper Sec. 5.3): this public registry keeps all baseline
# selection paths identical while each implementation remains in its own file.

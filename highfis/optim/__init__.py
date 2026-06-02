"""Training strategies for highFIS estimators.

Trainers decouple the optimisation loop from the sklearn estimator layer.
GradientTrainer implements standard single-phase mini-batch gradient
descent. DGTrainer implements the three-phase data-guided (DG) training
protocol required by DG-TSK and DG-ALETSK. FSRETrainer implements the
three-phase FSRE training protocol required by FSRE-ADATSK.

Optimisers:
    - `GradientTrainer` — standard mini-batch gradient descent.
    - `DGTrainer` — data-guided three-phase protocol for DG-TSK and DG-ALETSK.
    - `FSRETrainer` — three-phase FSRE protocol for FSRE-ADATSK.
"""

from ._base import BaseTrainer
from ._dg import DGTrainer
from ._fsre import FSRETrainer
from ._gradient import GradientTrainer

__all__ = ["BaseTrainer", "DGTrainer", "FSRETrainer", "GradientTrainer"]

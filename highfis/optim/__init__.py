"""Training strategies for highFIS estimators.

Trainers decouple the optimisation loop from the sklearn estimator layer.
The :class:`GradientTrainer` implements standard single-phase mini-batch
gradient descent.  The :class:`DGTrainer` implements the three-phase
data-guided (DG) training protocol required by DG-TSK and DG-ALETSK.
The :class:`FSRETrainer` implements the three-phase FSRE training protocol
required by FSRE-ADATSK.

Example::

    from highfis import DGTSKClassifier
    from highfis.optim import DGTrainer

    clf = DGTSKClassifier(trainer=DGTrainer(dg_epochs=20, finetune_epochs=100))
    clf.fit(X_train, y_train, x_val=X_val, y_val=y_val)
"""

from ._base import BaseTrainer
from ._dg import DGTrainer
from ._fsre import FSRETrainer
from ._gradient import GradientTrainer

__all__ = ["BaseTrainer", "DGTrainer", "FSRETrainer", "GradientTrainer"]

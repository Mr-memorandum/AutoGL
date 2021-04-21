__version__ = "0.1.1"

# before everything setup, we need to determine backend
from .backend import _check_backend
_check_backend()

from . import backend, data, datasets, solver, utils
from .module import train, model, feature, hpo, ensemble

__all__ = ['backend', 'data', 'datasets', 'solver', 'utils', 'train', 'model', 'feature', 'feature', 'hpo', 'ensemble']

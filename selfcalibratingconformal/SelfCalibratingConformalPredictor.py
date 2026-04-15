import sys
from types import ModuleType

from selfcalibratingconformal.regression import SelfCalibratingConformalPredictor


class _CallableModule(ModuleType):
    def __call__(self, *args, **kwargs):
        return SelfCalibratingConformalPredictor(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["SelfCalibratingConformalPredictor"]

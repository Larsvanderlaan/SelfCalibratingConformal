import sys
from types import ModuleType

from selfcalibratingconformal.quantile import VennAbersQuantileConformalPredictor


class _CallableModule(ModuleType):
    def __call__(self, *args, **kwargs):
        return VennAbersQuantileConformalPredictor(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["VennAbersQuantileConformalPredictor"]

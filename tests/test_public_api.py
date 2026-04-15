from selfcalibratingconformal import (
    SelfCalibratingConformalPredictor,
    VennAbersQuantileConformalPredictor,
)
from selfcalibratingconformal.SelfCalibratingConformalPredictor import (
    SelfCalibratingConformalPredictor as LegacyRegressionImport,
)
from selfcalibratingconformal.VennAbersQuantileConformalPredictor import (
    VennAbersQuantileConformalPredictor as LegacyQuantileImport,
)


def test_public_imports_remain_available():
    assert SelfCalibratingConformalPredictor is LegacyRegressionImport
    assert VennAbersQuantileConformalPredictor is LegacyQuantileImport

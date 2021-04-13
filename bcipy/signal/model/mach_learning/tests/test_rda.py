import numpy as np
from bcipy.signal.model.mach_learning.classifier import RegularizedDiscriminantAnalysis
from numpy.testing import assert_allclose


def test_rda():
    dim_x = 2
    num_x_p = 2000
    num_x_n = 500

    x_p = 2 * np.random.randn(num_x_p, dim_x)
    x_n = 4 + np.random.randn(num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    x = np.concatenate(np.asarray([x_p, x_n]), 0)
    y = np.concatenate(np.asarray([y_p, y_n]), 0)

    rda = RegularizedDiscriminantAnalysis()

    z = rda.fit_transform(x, y)
    rda.fit(x, y)
    z_2 = rda.transform(x)
    assert_allclose(z, z_2)

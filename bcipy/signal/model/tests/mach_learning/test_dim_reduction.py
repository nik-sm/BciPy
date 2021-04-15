from bcipy.signal.model.mach_learning.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
import numpy as np
from numpy.testing import assert_allclose


def test_channelwise_pca():
    np.random.seed(0)

    dim_x = 10
    num_ch = 8
    num_x_p = 500
    num_x_n = 200
    var_tol = 0.95

    # We only require the data not labels
    x_p = 2 * np.random.randn(num_ch, num_x_p, dim_x)
    x_n = np.random.randn(num_ch, num_x_n, dim_x)
    x = np.concatenate((x_n, x_p), axis=1)

    cw_pca = ChannelWisePrincipalComponentAnalysis(num_ch=num_ch)
    y = cw_pca.fit_transform(x, var_tol=var_tol)
    y_2 = cw_pca.transform(x)
    # TODO - the demo mentioned MSE, but y and y_2 are identical by construction.
    # What is the point of comparing them?
    assert_allclose(y, y_2)
    assert y.shape == ((700, 20))

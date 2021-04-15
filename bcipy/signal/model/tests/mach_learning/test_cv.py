import numpy as np
from bcipy.signal.model.ml.cross_validation import cross_validation
from bcipy.signal.model.ml.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.ml.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.ml.pipeline import Pipeline

from numpy.testing import assert_almost_equal


def test_cv():
    np.random.seed(0)

    dim_x = 5
    num_ch = 8
    num_x_p = 500
    num_x_n = 200
    var_tol = 0.5

    x_p = 2 * np.random.randn(num_ch, num_x_p, dim_x)
    x_n = 4 + np.random.randn(num_ch, num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    x = np.concatenate((x_p, x_n), 1)
    y = np.concatenate(np.asarray([y_p, y_n]), 0)
    permutation = np.random.permutation(x.shape[1])
    x = x[:, permutation, :]
    y = y[permutation]

    rda = RegularizedDiscriminantAnalysis()
    pca = ChannelWisePrincipalComponentAnalysis(num_ch=num_ch, var_tol=var_tol)
    pipeline = Pipeline()
    pipeline.add(pca)
    pipeline.add(rda)
    # see bcipy.signal.model.mach_learning.train_model:44 where the return of cross_validation()
    # is broken into two named parts, "lam" and "gam".
    # TODO - these need better names and documentation! i.e. what is lambda and what is gamma?
    lam, gam = cross_validation(x, y, pipeline)

    assert_almost_equal(lam, 0.9)
    assert_almost_equal(gam, 0.1)

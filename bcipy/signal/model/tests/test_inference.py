""" Inference demo """

import matplotlib.pylab as plt
import numpy as np
from bcipy.signal.model.inference import inference
from bcipy.signal.model.mach_learning.train_model import train_pca_rda_kde_model

from string import ascii_uppercase
import pytest


@pytest.mark.mpl_image_compare()
def test_inference():
    np.random.seed(0)

    dim_x = 5
    num_ch = 8
    num_x_p = 100
    num_x_n = 900

    mean_pos = 0.8
    var_pos = 0.5
    mean_neg = 0
    var_neg = 0.5

    x_p = mean_pos + var_pos * np.random.randn(num_ch, num_x_p, dim_x)
    x_n = mean_neg + var_neg * np.random.randn(num_ch, num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    x = np.concatenate((x_p, x_n), 1)
    y = np.concatenate(np.asarray([y_p, y_n]), 0)
    permutation = np.random.permutation(x.shape[1])
    x = x[:, permutation, :]
    y = y[permutation]

    model, _ = train_pca_rda_kde_model(x, y, k_folds=10)

    alp = list(ascii_uppercase) + ["<", "_"]

    num_x_p = 1
    num_x_n = 9

    x_p_s = mean_pos + var_pos * np.random.randn(num_ch, num_x_p, dim_x)
    x_n_s = mean_neg + var_neg * np.random.randn(num_ch, num_x_n, dim_x)
    x_s = np.concatenate((x_n_s, x_p_s), 1)

    idx_let = np.random.permutation(len(alp))
    letters = [alp[i] for i in idx_let[0 : (num_x_p + num_x_n)]]

    lik_r = inference(x=x_s, targets=letters, model=model, alphabet=alp)

    fig, ax = plt.subplots()
    ax.plot(np.array(list(range(len(alp)))), lik_r, "ro")
    ax.set_xticks(np.arange(len(alp)))
    ax.set_xticklabels(alp)
    return fig

import numpy as np

from midas.weights import ExpAlmonWeights, BetaWeights


def test_beta_es():
    w = BetaWeights(1., 5.).weights(3)

    assert np.allclose(w, [0.941176, 0.0588238, 9.4118e-25])


def test_beta_es_nz():
    w = BetaWeights(1, 5, 0.1).weights(3)

    assert np.allclose(w, [0.800905, 0.122172, 0.076923])


def test_almon():
    w = ExpAlmonWeights(0.01, -0.0025).weights(3)

    assert np.allclose(w, [0.333055, 0.333889, 0.333055])


def test_x_weighted():
    x = np.ones((3, 3))
    bw = BetaWeights(1., 5.)

    xw, w = bw.x_weighted(x, [1., 5.])

    assert x.shape[0] == xw.shape[0]
    assert np.allclose(xw, [1., 1., 1.])

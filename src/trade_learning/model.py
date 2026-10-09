"""Public constants and admissible actions for the specified synthetic market."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.special import expit

K = np.array([[.75, .20, .05], [.10, .80, .10], [.05, .20, .75]], dtype=float)
ACTIONS = np.array(list(itertools.product((-1, 0, 1), repeat=2)) + [(-2, -2)] * 2,
                   dtype=np.int8)
K.setflags(write=False)
ACTIONS.setflags(write=False)
H = .025
DEPTH = .025
CP = .001
CT = .002
LAMBDA = .002
MU = .03
SIGMA = .30


def admissible(q, qmax=5):
    """Mask all fill-subset-safe actions, broadcasting over integer inventory.

    At the upper boundary *any* submitted bid is inadmissible, including a
    paired quote whose ask would offset it if that ask also filled.
    """
    q = np.asarray(q)
    if int(qmax) != qmax or qmax < 1:
        raise ValueError("qmax must be a positive integer")
    if not np.all(np.isfinite(q)) or np.any(q != np.floor(q)) or np.any(abs(q) > qmax):
        raise ValueError("inventory must be an integer within the configured bound")
    result = np.ones(q.shape + (11,), dtype=bool)
    result[..., :9] &= ~((q[..., None] >= qmax) & (ACTIONS[:9, 0] >= 0))
    result[..., :9] &= ~((q[..., None] <= -qmax) & (ACTIONS[:9, 1] >= 0))
    result[..., 9] = q < qmax
    result[..., 10] = q > -qmax
    return result


def fill_probability(x, side, depth):
    """Specified marginal fill probability, with ordinary NumPy broadcasting."""
    return expit(-.3 - .7 * np.asarray(depth) - .2 * np.asarray(side) * np.asarray(x))

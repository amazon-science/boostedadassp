# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import scipy
import scipy.linalg
from scipy.optimize import minimize
from scipy.stats import norm

from sklearn import preprocessing


def delta_eps(eps, mu):
    """Delta computation based on mu and epsilon.

     .. math::

        \begin{aligned}
            \delta(\epsilon) = \Phi(-\epsilon / \mu + \mu / 2) - \exp(\epsilon)\Phi(-\epsilon / \mu - \mu / 2)
        \end{aligned}


    Args:
        mu (float): privacy parameter in Gaussian Differential Privacy
        eps (float): privacy parameter in Approximate Differential Privacy

    Returns:
        delta (float): converted delta in Approximate Differential Privacy

    """
    delta = norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
    return delta

def convert_ApproxDP_to_GDP(eps: float, delta: float = 1e-6):
    """Convert the privacy parameters eps and delta in Approximate DP to the privacy parameter mu in Gaussian DP

    With the same privacy loss, Gaussian DP allows more interactions with the data than Approximate DP does.
    The underlying composition over multiple campaigns is done through Gaussian DP.

    Once we receive the total privacy budget in eps provided by a customer, this function converts (eps, delta) pair to mu.

    Args:
        eps (float): privacy parameter in Approximate Differential Privacy
        delta (float): privacy parameter in Approximate Differential Privacy

    Returns:
        mu (float): privacy parameter in Gaussian Differential Privacy
    """

    assert eps > 0
    assert delta > 0

    res = minimize(
        fun=lambda mu: (np.log(delta_eps(eps, mu)) - np.log(delta)) ** 2.0,
        x0=eps,
        bounds=((delta, None),),
        tol=delta**2.0,
        method="Nelder-Mead",
        options={"maxiter": 10000},
    )
    mu = res.x

    return mu

class BoostedAdaSSP:
    def __init__(
        self,
        x_bound=1,
        y_bound=1,
        epsilon=1,
        delta=1e-6,
        num_iterations=100,
        shrinkage="constant",
        random_state=np.random.RandomState(42),
    ):
        self.rng = random_state
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.epsilon = epsilon
        self.delta = delta
        self.num_iterations = num_iterations

        if shrinkage == "constant":
            self.shrinkage = lambda x: 1
        if shrinkage == "1/T":
            self.shrinkage = lambda x: 1/x
        if shrinkage == "1/T**0.5":
            self.shrinkage = lambda x: 1/x ** 0.5
       
        self.sigma = convert_ApproxDP_to_GDP(self.epsilon, self.delta)


    def clipping_norm(self, X):
        normalized_X = preprocessing.normalize(X, norm="l2")
        length_X = np.linalg.norm(X, axis=1, keepdims=True)
        clipped_X = normalized_X * length_X.clip(min=0, max=self.x_bound)

        return clipped_X

    def noisy_cov(self, XTX):
        # GM1
        Z = self.x_bound**2 * self.sigma * self.rng.normal(size=XTX.shape)

        Z_analyzegauss = np.triu(Z) + np.triu(Z, k=1).T
        hatXTX = XTX + Z_analyzegauss
        # GM3
        s = scipy.linalg.eigvalsh(XTX, subset_by_value=(0, np.inf))
        s = s[::-1]

        lambdamin = s[-1] + self.x_bound**2 * self.sigma * self.rng.normal(size=1)
        lambdamin_lowerbound = max(0, lambdamin - self.x_bound**2 * self.sigma * 1.96)

        dim = XTX.shape[0]
        lamb = max(
            0,
            np.sqrt(dim) * self.sigma * self.x_bound**2 * 1.96 - lambdamin_lowerbound,
        )

        return hatXTX + lamb * np.eye(dim)

    def run_AdaSSP(self, hatXTX, XTy):
        # GM2
        hatXTy = XTy + self.sigma * self.x_bound * self.y_bound * self.rng.normal(
            size=XTy.shape
        )
        theta_adassp = scipy.linalg.solve(hatXTX, hatXTy, assume_a="sym")
        return theta_adassp

    def fit(self, X, y):
        X = self.clipping_norm(X)

        n, dim = X.shape

        XTX = X.T @ X 

        hatXTX = self.noisy_cov(XTX)

        ensemble_theta = np.zeros(dim)

        for i in range(self.num_iterations):
            residual = y - X @ ensemble_theta
            residual = residual.clip(-self.y_bound, self.y_bound)
            XTy = X.T @ residual 

            theta = self.run_AdaSSP(
                hatXTX,
                XTy,
            )

            shrinkage = self.shrinkage((i+1))
            ensemble_theta += shrinkage * theta

        self.ensemble_theta = ensemble_theta
        return self

    def predict(self, X):
        X = self.clipping_norm(X)
        return X @ self.ensemble_theta

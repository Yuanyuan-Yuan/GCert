#####################################################################
# This script provides the implementation of low-rank factorization #
# (a.k.a. robust PCA). The implementation is based on               #
# https://github.com/zhujiapeng/LowRankGAN/blob/master/RobustPCA.py #
#####################################################################

import numpy as np

class RobustPCA(object):

    def __init__(self, M, lamb=1/60):
        self.M = M
        self.S = np.zeros(self.M.shape)     # sparse matrix
        self.L = np.zeros(self.M.shape)     # low-rank matrix
        self.Lamb = np.zeros(self.M.shape)  # Lambda matrix
        # mu is the coefficient used in augmented Lagrangian.
        self.mu = np.prod(self.M.shape) / (4 * np.linalg.norm(self.M, ord=1))
        self.mu_inv = 1 / self.mu
        self.iter = 0
        self.error = 1e-7 * self.frobenius_norm(self.M)

        if lamb:
            self.lamb = lamb
        else:
            self.lamb = 1 / np.sqrt(np.max(self.M.shape))

    def reset_iter(self):
        """Resets the iteration."""
        self.iter = 0

    @staticmethod
    def frobenius_norm(M):
        """Computes the Frobenius norm of a given matrix."""
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, VH = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), VH))

    def fit(self, max_iter=10000, iter_print=100):
        self.reset_iter()
        err_i = np.Inf
        S_k = self.S
        L_k = self.L
        Lamb_k = self.Lamb

        while (err_i > self.error) and self.iter < max_iter:
            L_k = self.svd_threshold(
                self.M - S_k - self.mu_inv * Lamb_k, self.mu_inv)
            S_k = self.shrink(
                self.M - L_k - self.mu_inv * Lamb_k, self.mu_inv * self.lamb)
            Lamb_k = Lamb_k + self.mu * (L_k + S_k - self.M)
            err_i = self.frobenius_norm(L_k + S_k - self.M)
            self.iter += 1
            # if (self.iter % iter_print) == 0:
            #     print(f'iteration: {self.iter}, error: {err_i}')

        return L_k, S_k
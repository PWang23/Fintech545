import numpy as np
import random
import scipy.linalg as la

def chol_pd(a):
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        root[j, j] = np.sqrt(a[j, j] - s)
        ir = 1.0 / root[j, j]
        for i in range(j + 1, n):
            s = np.dot(root[i, :j], root[j, :j])
            root[i, j] = (a[i, j] - s) * ir
    return root

n = 5
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1.0)

root = np.zeros((n, n))
root = chol_pd(sigma)

np.allclose(root @ root.T, sigma)

root2 = la.cholesky(sigma, lower=True)
np.allclose(root, root2)

sigma[0, 1] = 1.0
sigma[1, 0] = 1.0

eigvals = np.linalg.eigvals(sigma)

root = chol_pd(sigma)

def chol_psd(a):
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        temp = a[j, j] - s
        if temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:
            root[j, j+1:] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    return root

root = chol_psd(sigma)

np.allclose(root @ root.T, sigma)

root2 = la.cholesky(sigma, lower=True)
np.allclose(root, root2)

sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

eigvals = np.linalg.eigvals(sigma)

root = chol_psd(sigma)


import numpy as np
from scipy.linalg import cholesky, eigvals
import random
import pandas as pd
import matplotlib.pyplot as plt

# Cholesky that assumes PD matrix
def chol_pd(a):
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = 0
        if j > 0:
            s = root[j, :j].dot(root[j, :j])
        root[j, j] = np.sqrt(a[j, j] - s)

        ir = 1 / root[j, j]
        for i in range(j + 1, n):
            s = root[i, :j].dot(root[j, :j])
            root[i, j] = (a[i, j] - s) * ir
    
    return root

n = 5
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1)

root = chol_pd(sigma)

assert np.allclose(root @ root.T, sigma)
root2 = cholesky(sigma, lower=True)
assert np.allclose(root, root2)

#make the matrix PSD
sigma[0, 1] = 1.0
sigma[1, 0] = 1.0
eigvals(sigma)



#Cholesky that assumes PSD
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

root2 = cholesky(sigma, lower=True)

#make the matrix slightly non-definite
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

eigvals = eigvals(sigma)

root = chol_psd(sigma)


#Generate some random numbers with missing values.
def generate_with_missing(n, m, pmiss=0.25):
    x = np.empty((n, m), dtype=object)

    for i in range(n):
        for j in range(m):
            if random.random() >= pmiss:
                x[i, j] = random.gauss(0, 1)
    return x

np.random.seed(2)
x = generate_with_missing(10, 5, pmiss=0.2)
np.cov(x)


#calculate either the covariance or correlation function when there are missing values
def missing_cov(x, skipMiss=True, fun=np.cov):
    n, m = x.shape
    nMiss = np.count_nonzero(np.isnan(x), axis=0)

    if np.sum(nMiss) == 0:
        return fun(x)

    idxMissing = []
    for j in range(m):
        idxMissing.append(np.where(np.isnan(x[:, j]))[0])

    if skipMiss:
        rows = set(range(n))
        for c in range(m):
            for rm in idxMissing[c]:
                rows.remove(rm)
        rows = np.sort(list(rows))
        return fun(x[rows, :])
    else:
        out = np.empty((m, m), dtype=float)
        for i in range(m):
            for j in range(i, m):
                rows = set(range(n))
                for c in (i, j):
                    for rm in idxMissing[c]:
                        rows.remove(rm)
                rows = np.sort(list(rows))
                out[i, j] = fun(x[rows, [i, j]])[0, 1]
                if i != j:
                    out[j, i] = out[i, j]
        return out

skipMiss = missing_cov(x)
pairwise = missing_cov(x, skipMiss=False)
eigvals(pairwise)

root = chol_psd(skipMiss)
root = chol_psd(pairwise)


#Look at Exponential Weights
def populateWeights(x, w, cw, λ):
    n = len(x)
    tw = 0.0
    for i in range(n):
        x[i] = i+1
        w[i] = (1-λ)*(λ**i)
        tw += w[i]
        cw[i] = tw
    for i in range(n):
        w[i] = w[i]/tw
        cw[i] = cw[i]/tw

weights = pd.DataFrame()
cumulative_weights = pd.DataFrame()
n=100
x = np.empty(n)
w = np.empty(n)
cumulative_w = np.empty(n)

#calculated weights λ=75%
populateWeights(x, w, cumulative_w, 0.75)
weights['x'] = x
weights['λ=0.75'] = w
cumulative_weights['x'] = x
cumulative_weights['λ=0.75'] = cumulative_w

#calculated weights λ=90%
populateWeights(x, w, cumulative_w, 0.90)
weights['λ=0.90'] = w
cumulative_weights['λ=0.90'] = cumulative_w

#calculated weights λ=97%
populateWeights(x, w, cumulative_w, 0.97)
weights['λ=0.97'] = w
cumulative_weights['λ=0.97'] = cumulative_w

#calculated weights λ=99%
populateWeights(x, w, cumulative_w, 0.99)
weights['λ=0.99'] = w
cumulative_weights['λ=0.99'] = cumulative_w

#plot Weights
for i in range(1, len(weights.columns)):
    plt.plot(weights['x'], weights.iloc[:, i])
plt.title("Weights")
plt.legend(weights.columns[1:])
plt.show()

#plot the cumulative weights
for i in range(1, len(cumulative_weights.columns)):
    plt.plot(cumulative_weights['x'], cumulative_weights.iloc[:, i])
plt.title("Cumulative Weights")
plt.legend(cumulative_weights.columns[1:])
plt.show()





#Near PSD Matrix
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = np.copy(a)

    #calculate the correlation matrix if we got a covariance
    if (np.diag(out) != 1.0).sum() != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    
    #SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals,epsilon)
    T = 1 / (vecs * vals * vecs)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    #Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out

near_pairwise = near_psd(pairwise)




#PCA
def simulate_pca(a, nsim, nval=None):
    #Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)

    #flip values and vectors
    flip = np.fliplr(np.arange(vals.shape[0]).reshape(-1,1))[0]
    vals = vals[flip]
    vecs = vecs[:,flip]
    
    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if nval is not None:
        if nval < posv.shape[0]:
            posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:,posv]

    print("Simulating with {} PC Factors: {:.2f}% total variance explained".format(posv.shape[0], sum(vals)/tv*100))
    B = vecs @ np.diag(np.sqrt(vals))

    m = vals.shape[0]
    r = np.random.randn(m, nsim)

    return (B @ r).T

n = 5
sigma = np.full((n,n), 0.9)
np.fill_diagonal(sigma, 1.0)

sigma[0,1]=1
sigma[1,0]=1

v = np.diag(np.full(n, 0.5))
sigma = v @ sigma @ v

sim = simulate_pca(sigma, 10000)
np.cov(sim.T)

sim = simulate_pca(sigma, 10000, nval=3)
np.cov(sim.T)

sim = simulate_pca(sigma, 10000, nval=2)
np.cov(sim.T)

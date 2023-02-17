import pandas as pd
import numpy as np

def return_calculate(prices: pd.DataFrame, method="DISCRETE", dateColumn="date"):
    vars = list(prices.columns)
    nVars = len(vars)
    vars.remove(dateColumn)
    vars = [str(var) for var in vars]
    if nVars == len(vars):
        raise ValueError("dateColumn: " + dateColumn + " not in DataFrame: " + str(vars))
    nVars = nVars - 1

    p = np.matrix(prices[vars])
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.empty((n-1,m), dtype=float)
    # p = p[0:(n-1),:]

    for i in range(0, n-1):
        for j in range(0, m):
            p2[i,j] = p[i+1,j] / p[i,j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError("method: " + method + " must be in (\"LOG\",\"DISCRETE\")")

    dates = prices[dateColumn][1:n]
    out = pd.DataFrame({dateColumn: dates})
    for i in range(0, nVars):
        out[vars[i]] = p2[:,i]
    return out

import numpy as np

def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]
    return temp

def SSA(lb, ub, dim, fun, pop=50, M=200):
    P_percent = 0.4
    W_percent = 0.2
    pNum = round(pop * P_percent)
    wNum = round(pop * W_percent)
    X = np.zeros((pop, dim), dtype="int32")
    fit = np.zeros((pop, 1))
    for i in range(pop):
        X[i, :] = np.int32(lb + (ub - lb) * np.random.rand(1, dim))
        fit[i, 0] = fun(X[i, :])
    pFit = fit
    pX = X
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]

    for t in range(M):
        sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0])
        worse = X[B, :]
        r2 = np.random.rand(1)

        if r2 < 0.8:
            for i in range(pNum):
                r1 = np.random.rand(1)
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * np.exp(-(i) / (r1 * M))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        elif r2 >= 0.8:
            for i in range(pNum):
                Q = np.random.rand(1)
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] + 5 * Q * np.ones((1, dim))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :]

        for ii in range(pop - pNum):
            i = ii + pNum
            A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1
            if i > (pop - pNum) / 2 + pNum:
                Q = np.random.rand(1)
                X[sortIndex[0, i], :] = Q * np.exp(worse - pX[sortIndex[0, i], :] / np.square(i))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            else:
                X[sortIndex[0, i], :] = bestXX + np.dot(np.abs(pX[sortIndex[0, i], :] - bestXX),
                                                        1 / (A.T * np.dot(A, A.T))) * np.ones((1, dim))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])

        arrc = np.arange(len(sortIndex[0, :]))
        c = np.random.permutation(arrc)
        b = sortIndex[0, c[0:wNum]]

        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:
                X[sortIndex[0, b[j]], :] = bestX + np.random.rand(1, dim) * np.abs(pX[sortIndex[0, b[j]], :] - bestX)
            else:
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :] + (2 * np.random.rand(1) - 1) * np.abs(
                    pX[sortIndex[0, b[j]], :] - worse) / (pFit[sortIndex[0, b[j]]] - fmax + 10 ** (-50))
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)
            fit[sortIndex[0, b[j]], 0] = fun(X[sortIndex[0, b[j]]])

        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]


    return fMin, bestX,

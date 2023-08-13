import sympy as sym
import numpy as np
l = x, y = sym.symbols('x y')
f = 2 * x**2 + 2 * x * y + y**2 + x - y
l3 = [i for i in range(len(l))]
def GD1(x0, eta, n):
    X0 = np.array(x0)
    l1 = [0 for i in range(len(l))]
    X = np.array([l1 for i in range(n + 1)], dtype = float)
    X[0] = X0
    for i in range(1, n + 1):
        l2 = [(l[j], X[i - 1][j]) for j in range(len(l))]
        l4 = X[i - 1] - eta * np.array([sym.diff(f, j).subs(l2) for j in l])
        np.copyto(X[i], l4, casting = 'unsafe')
    return X[n]
def GD2(x0, eta, epsilon):
    X0 = np.array(x0)
    l1 = np.array([0 for i in range(len(l))], dtype = float)
    X = np.array([l1])
    X[0] = X0
    i = 1
    while(True):
        l2 = [(l[j], X[i - 1][j]) for j in range(len(l))]
        l4 = X[i - 1] - eta * np.array([sym.diff(f, j).subs(l2) for j in l])
        X = np.concatenate((X, np.array([l1])))
        np.copyto(X[i], l4, casting = 'unsafe')
        l2 = [(l[j], X[i][j]) for j in range(len(l))]
        if(np.linalg.norm(np.array([float(sym.diff(f, j).subs(l2)) for j in l])) < epsilon):
            break
        i = i + 1
    return X[i]
def GD3(x0, eta, epsilon):
    X0 = np.array(x0)
    l1 = np.array([0 for i in range(len(l))], dtype = float)
    X = np.array([l1])
    X[0] = X0
    i = 1
    while(True):
        l2 = [(l[j], X[i - 1][j]) for j in range(len(l))]
        l4 = X[i - 1] - eta * np.array([sym.diff(f, j).subs(l2) for j in l])
        X = np.concatenate((X, np.array([l1])))
        l2 = [(l[j], X[i][j]) for j in range(len(l))]
        np.copyto(X[i], l4, casting = 'unsafe')
        if(np.linalg.norm(X[i] - X[i - 1]) < epsilon):
            break
        i = i + 1
    return X[i]
def GD4(x0, eta, epsilon):
    X0 = np.array(x0)
    l1 = np.array([0 for i in range(len(l))], dtype = float)
    X = np.array([l1])
    X[0] = X0
    i = 1
    while(True):
        l2 = [(l[j], X[i - 1][j]) for j in range(len(l))]
        l4 = X[i - 1] - eta * np.array([sym.diff(f, j).subs(l2) for j in l])
        X = np.concatenate((X, np.array([l1])))
        l2 = [(l[j], X[i][j]) for j in range(len(l))]
        np.copyto(X[i], l4, casting = 'unsafe')
        if(np.linalg.norm(X[i] - X[i - 1]) / (np.linalg.norm(X[i - 1])) < epsilon):
            break
        i = i + 1
    return X[i]
print(GD1([-1, 1.8], 0.1, 200))
print(GD2([-1, 1.8], 0.1, 0.00000001))
print(GD3([-1, 1.8], 0.1, 0.00000001))
print(GD4([-1, 1.8], 0.1, 0.00000001))
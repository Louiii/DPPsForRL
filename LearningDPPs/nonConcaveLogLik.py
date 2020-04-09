import numpy as np
from numpy.linalg import det, inv
import matplotlib.pyplot as plt

def L(eta):
	K = K1 + eta * K2
	return np.dot(K, inv(np.eye(N) - K))

def L_At(At, eta):
	return L(eta)[np.ix_(At, At)]

def logLikelihood(A, eta):
	T = len(A)
	term1 = -T*np.log(det( L(eta) + np.eye(N) ))
	term2 = np.sum([np.log(det( L_At(At, eta) )) for At in A])
	return term1 + term2

N = 2
K1 = np.array([[0.5, 0],[0, 0.4]])
K2 = np.array([[0.1, 0.6],[0.6, -0.1]])

A = [[0,1], [0], [1]]
x = list(np.linspace(0, 0.5, 50))
y = [logLikelihood(A, xi) for xi in x]

plt.plot(x, y, c='turquoise')
plt.xlabel('$\\eta$')
plt.ylabel('log-likelihood')
plt.savefig('plots/nonConcave', dpi=400)
plt.show()
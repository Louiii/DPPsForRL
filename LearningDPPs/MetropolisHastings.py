import sys
sys.path.append('../')
from DPP import DPP
from scipy.linalg import eigh
from tqdm import tqdm
import numpy as np
from numpy.linalg import det
import pylab
import matplotlib.pyplot as plt

def L(theta):
	qs = np.array([ q(i, theta) for i in baseY])
	return np.multiply( S, np.outer( qs, qs ) )

def L_At(At, theta):# L(At|theta)
	qs = np.array([ q(i, theta) for i in At])
	return np.multiply( S[np.ix_(At, At)], np.outer( qs, qs ) )

def logLikelihood(A, theta):# log likelihood function for a standard DPP (not k-DPP!)
	T = len(A)
	term1 = -T*np.log(det( L(theta) + np.eye(N) ))
	term2 = np.sum([np.log(det( L_At(At, theta) )) for At in A])
	return term1 + term2

n, sigma = 20, 0.2
N = n*n
x, y = np.meshgrid(np.linspace(1/n, 1, n), np.linspace(1/n, 1, n))
xf, yf = x.flatten('F'), y.flatten('F')
baseY = list(range(N))

stack_xf, stack_yf = np.array([xf,]*N), np.array([yf,]*N)
M = np.square( (stack_xf.T - stack_xf) ) + np.square( (stack_yf.T - stack_yf) )
S = np.exp(-M/sigma**2)


fx = lambda i: np.array([ xf[i], yf[i] ])
q = lambda i, theta: np.dot( fx(i), theta )

theta_true, T = np.array([4.1, 0.76]), 200
dpp = DPP(*eigh(L(theta_true)))
print('...generating samples')
samples = [dpp.sample() for _ in tqdm(range(T))]




unif = np.random.uniform

T = 10000

f = lambda th: logLikelihood(samples, th)
g = lambda x: np.random.normal(x, 0.1)#np.array([unif(2, 5), unif(0.5, 1)])
# g is our proposal (or prior) because we consider only consider the ratio
# of g(z|xt) to g(xt|z) and because the normal dist. is symmetric we can 
# ignore it from the ratio.

def r(th1, th2):
	logdiff = logLikelihood(samples, th1) - logLikelihood(samples, th2)
	return np.exp(logdiff)

def MH(x0):
	x = []
	xt = x0
	accept = 0
	for i in tqdm(range(T)):
		z = g(xt)
		alpha = min(r(z, xt), 1)
		if unif() < alpha:
			xt = z
			accept += 1
		x.append(xt)
	print('Acceptance rate: '+str(accept/T))
	return x

x0 = np.array([8, 3])
samples = MH(x0)
x, y = zip(*samples)

plt.hist2d(x[200:], y[200:], bins=26)
plt.colorbar()
plt.savefig('plots/mh_hist', dpi=400)
plt.show()
plt.clf()

pylab.plot(x[-10000:], y[-10000:], lw=0.4, c='c') 
pylab.savefig('plots/random_walk', dpi=400)
pylab.show() 
import sys
sys.path.append('../')
from DPP import DPP
from scipy.linalg import eigh
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np
from numpy.linalg import det
import pylab
import matplotlib.pyplot as plt

def L(theta):
	qs = np.array([ q(i, theta) for i in baseY])
	return np.multiply( K, np.outer( qs, qs ) )

def L_At(At, theta):# L(At|theta)
	qs = np.array([ q(i, theta) for i in At])
	return np.multiply( K[np.ix_(At, At)], np.outer( qs, qs ) )

def logLikelihood(A, theta):# log likelihood function for a standard DPP (not k-DPP!)
	T = len(A)
	term1 = -T*np.log(det( L(theta) + np.eye(N) ))
	term2 = np.sum([np.log(det( L_At(At, theta) )) for At in A])
	return term1 + term2

def lk_ratio(th1, th2, A):
	T = len(A)
	fctr = det( L(th2) + np.eye(N) ) / det( L(th1) + np.eye(N) )
	prod = 1
	for At in A:
		print(prod)
		prod *= fctr * det( L_At(At, th1) ) / det( L_At(At, th2) )
	return prod

# def likelihood(A, theta):
# 	T = len(A)
# 	denom = det( L(theta) + np.eye(N) )
# 	prod = 1
# 	for At in A:
# 		print(prod)
# 		prod *= det( L_At(At, theta) ) / denom
# 	return prod


n, sigma = 20, 0.2
N = n*n
x, y = np.meshgrid(np.linspace(1/n, 1, n), np.linspace(1/n, 1, n))
xf, yf = x.flatten('F'), y.flatten('F')
baseY = list(range(N))

stack_xf, stack_yf = np.array([xf,]*N), np.array([yf,]*N)
M = np.square( (stack_xf.T - stack_xf) ) + np.square( (stack_yf.T - stack_yf) )
K = np.exp(-M/sigma**2)


fx = lambda i: np.array([ xf[i], yf[i] ])
q = lambda i, theta: np.dot( fx(i), theta )

theta_true, T = np.array([4.1, 0.76]), 100
dpp = DPP(*eigh(L(theta_true)))
print('...generating samples')
samples = [dpp.sample() for _ in tqdm(range(T))]






unif = np.random.uniform

T = 1000

f = lambda th: logLikelihood(samples, th)
g = lambda x: np.random.normal(x, 0.1)#np.array([unif(2, 5), unif(0.5, 1)])

r = lambda th1,th2: lk_ratio(th1, th2, samples)


x0 = np.array([4, 0.8])

x = []
xt = x0
accept = 0
for i in tqdm(range(T)):
	z = g(xt)
	# print('likelihoods:  '+str(f(z))+', '+str(f(xt)))
	# alpha = min(f(z)/f(xt), 1)
	alpha = min(r(z,xt), 1)
	u = unif()
	if u < alpha:
		x.append(z)
		xt = z
		accept += 1
	else:
		x.append(xt)

x, y = zip(*x)

print('Acceptance rate: '+str(accept/T))
plt.hist2d(x, y, bins=20)
plt.show()

pylab.title("Random Walk ($n = " + str(T) + "$ steps)") 
pylab.plot(x[-10000:], y[-10000:], lw=0.4, c='c') 
pylab.show() 
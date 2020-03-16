import sys
sys.path.append('../')
from DPP import DPP
from scipy.linalg import eigh
from tqdm import tqdm
import numpy as np


f = lambda i: np.array([ xf[i], yf[i] ])
q = lambda i, theta: np.dot( f(i), theta )

def L(theta):
	qs = np.array([ q(i, theta) for i in baseY])
	return np.multiply( K, np.outer( qs, qs ) )

def L_At(At, theta):# L(At|theta)
	qs = np.array([ q(i, theta) for i in At])
	return np.multiply( K[np.ix_(At, At)], np.outer( qs, qs ) )

def dL_dtheta_l(l, theta):
    ''' The gradient of L with repect to the l^th parameter in theta '''
    fs = np.array([ f(i)[l] for i in baseY])
    qs = np.array([ q(i, theta) for i in baseY])
    fq = np.outer( fs, qs )
    return np.multiply( K, fq+fq.T )

def dL_At_dtheta_l(At, l, theta):
    ''' The gradient of L_At with repect to the l^th parameter in theta '''
    fs = np.array([ f(i)[l] for i in At])
    qs = np.array([ q(i, theta) for i in At])
    fq = np.outer( fs, qs )
    return np.multiply( K[np.ix_(At, At)], fq+fq.T )

def dLikelihood_dtheta_l(theta, A, l):
    ''' Gradient of the likelihood wrt theta_l'''
    term1 = np.sum([np.trace(np.dot(np.linalg.inv(L_At(At, theta)), dL_At_dtheta_l(At, l, theta))) for At in A])
    term2 = len(A)*np.trace(np.dot(np.linalg.inv(L(theta)+np.eye(N)), dL_dtheta_l(l, theta)))
    return term1-term2

def dLhd_dtheta(theta, A):
    ''' Gradient of the likelihood wrt theta, 'A' is an array of samples '''
    return np.array([dLikelihood_dtheta_l(theta, A, l) for l in range(len(theta))])

def SGD(theta, A, m, batch_size=10, decay_start=20, alpha_0 = 1):
    grad = []
    for t in range(1, m+1):
        alpha = alpha_0 * min(1, (decay_start/t)**2)
        As = A[np.random.randint(0, len(A), batch_size)]
        dth = dLhd_dtheta(theta, As)
        dth /= np.linalg.norm(dth)
        theta += alpha * dth
        grad.append((theta.copy(), dth.copy()))
        print((theta, t, alpha))
    return theta, grad

def Adam(theta, A, n_iter, alpha=0.5, batch_size=10, beta_1=0.9, beta_2=0.999, epsilon=0.001):
    m, v = np.zeros(len(theta))
    grad = []
    for t in range(1, n_iter+1):
        As = A[np.random.randint(0, len(A), batch_size)]
        g = dLhd_dtheta(theta, As)
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - np.power(beta_1, t))
        v_hat = v / (1 - np.power(beta_2, t))
        theta += alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        grad.append((theta.copy(), (g/np.linalg.norm(g)).copy()))
        print((theta, t, alpha))
    return theta, grad

n, sigma = 20, 0.2
N = n*n
x, y = np.meshgrid(np.linspace(1/n, 1, n), np.linspace(1/n, 1, n))
xf, yf = x.flatten('F'), y.flatten('F')
baseY = list(range(N))

stack_xf, stack_yf = np.array([xf,]*N), np.array([yf,]*N)
M = np.square( (stack_xf.T - stack_xf) ) + np.square( (stack_yf.T - stack_yf) )
K = np.exp(-M/sigma**2)



theta_true, T = np.array([4.1, 0.76]), 200
dpp = DPP( *eigh( L(theta_true) ) )
print('...generating samples')
samples = [dpp.sample() for _ in tqdm(range(T))]


# print('SGD:')
# theta = np.array([10.,10.])
# theta, grad = SGD(theta, np.array(samples), 100)
print('Adam:')
theta = np.array([10.,10.])
theta, grad = Adam(theta, np.array(samples), 200)


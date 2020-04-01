from misc import makeGridGraph, plotter
from scipy.linalg import eigh
import networkx as nx
import numpy as np
import time

from AldousBroder import AldousBroder
from Wilson import Wilson
from HKPV import USG

import sys
sys.path.append('../../')
from DPP import DPP


class SpanningGraphs:
    def __init__(self, n):
        self.g, self.pos = makeGridGraph(n)

    def aldousBroder(self):
        ab = AldousBroder(self.g)
        return ab.random_walk()

    def wilson(self):
        wn = Wilson(self.g)
        return wn.sample()

    def compute_kernel(self):
        inc_mat = nx.incidence_matrix(self.g, oriented=False)
        A = inc_mat[1:, :].toarray()
        #K=A^T[AA^T]^{-1}A
        self.K = np.dot(A.T, np.dot(np.linalg.inv(np.dot(A, A.T)), A))
        self.edges = list(self.g.edges())

    def dpp(self):
        self.compute_kernel()
        dpp_ = DPP(*eigh(self.K))
        Y = dpp_.sample(k=len(self.g.nodes)-1)

        sampl = nx.Graph()
        sampl.add_edges_from([self.edges[e] for e in Y])
        return sampl

    def HKPV(self):
        usg = USG(self.g)
        return usg.sample()


n = 10
sg = SpanningGraphs(n)

for fn in ["wilson", "aldousBroder", "dpp", "HKPV"]:
    t0 = time.time()
    object = getattr(sg, fn)
    sampl = object()
    t1 = time.time()
    print("Time to sample ",str(fn),": ", str(t1-t0))

    plotter(sampl, sg.pos, title=fn, nodes_on=False)
    

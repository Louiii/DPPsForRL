from misc import makeGridGraph, plotter
from scipy.linalg import qr
import networkx as nx
import numpy as np
import time

from AldousBroder import AldousBroder
from Wilson import Wilson
from HKPV import USG
import sys
sys.path.append('../')
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
        A = inc_mat[:-1, :].toarray()
        self.kernel_eig_vecs, _ = qr(A.T, mode='economic')
        self.edges = list(self.g.edges())

    def dpp(self):
        self.compute_kernel()
        evs = np.copy(self.kernel_eig_vecs)
        dpp = DPP(evs)
        dpp_sample = dpp.sample()

        sampl = nx.Graph()
        sampl.add_edges_from([self.edges[e] for e in dpp_sample])

        return sampl

    def HKPV(self):
        usg = USG(self.g)
        return usg.sample()


n = 15
sg = SpanningGraphs(n)

for fn in ["wilson", "aldousBroder", "dpp", "HKPV"]:
    t0 = time.time()
    object = getattr(sg, fn)
    sampl = object()
    t1 = time.time()
    print("Time to sample ",str(fn),": ", str(t1-t0))

    plotter(sampl, sg.pos, title=fn, nodes_on=False)
    

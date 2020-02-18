import networkx as nx
import numpy as np


class Wilson:
	def __init__(self, graph):
		''' Algorithm complexity: O(τ) 
			τ = mean hitting time (worst case is still O(mn))
		'''
		self.g = graph

	def randomTreeWithRoot(self, r):
		n = len(self.g.nodes())
		inTree = np.zeros(n, dtype=bool)
		Next   = np.zeros(n, dtype=int)
		inTree[r] = True
		for i in range(n):
			u = i
			while not inTree[u]:
				Next[u] = np.random.choice( list(self.g.neighbors(u)) )
				u = Next[u]
			u = i
			while not inTree[u]:
				inTree[u] = True
				u = Next[u]
		return Next, r

	def sample(self):
		start = np.random.choice(self.g.nodes())
		Next, r  = self.randomTreeWithRoot(start)
		edges = [(u, Next[u]) for u in self.g.nodes() if u != r]
		g = nx.Graph()
		g.add_nodes_from(self.g.nodes())
		g.add_edges_from(edges)
		return g
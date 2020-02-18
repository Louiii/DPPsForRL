import networkx as nx
import numpy as np


class AldousBroder:
	def __init__(self, graph):
		''' Uniform Random Spanning Tree: The Random Walk Algorithm
			Run a simple random walk started from an arbitrary vertex u in G until covering all vertices.
			For every vertex v = u, add the edge that we used the first time that we reached v to T.
			Return T 

			Algorithm complexity: O(mn) 
			n = num of vertices, m = num of edges
		'''
		self.g = graph

	def random_walk(self):
		start = np.random.choice(self.g.nodes())
		u = start
		visited = set([u])
		node_edge = { v : None for v in self.g.nodes()}
		n = len(self.g.nodes())
		while len(visited) < n:
			v = np.random.choice( list(self.g.neighbors(u)) )
			if v not in visited:
				visited.add(v)
				node_edge[v] = (u, v)
			u = v
		nodes = node_edge.keys()
		del node_edge[start]
		return self.make_graph(nodes, list(node_edge.values()))

	def make_graph(self, nodes, edges):
		g = nx.Graph()
		g.add_nodes_from(nodes)
		g.add_edges_from(edges)
		return g
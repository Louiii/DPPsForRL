import networkx as nx
# for some reason the functions imported in the next line dont import for me, if they 
# dont for you, they can be found online and copy and pasted into this file
from networkx import _count_lu_permutations, _laplacian_submatrix, resistance_distance
import numpy as np
from tqdm import tqdm

class USG:
    def __init__(self, g):
        self.fully_connected_graph = g
        self.all_edges, self.all_nodes = list(g.edges()), list(g.nodes())
        self.n = len(self.all_nodes)

    def contract_edge(self, u, v):
        new_vertex = self.n + len(self.tree)
        self.tree.update({ new_vertex : [u, v] } )
        return new_vertex

    def recurse(self, done, queue, v):
        if v not in self.tree:
            if len(queue)==0:
                return done + [v]
            w = queue.pop(0)
            return self.recurse(done+[v], queue, w)
        ws = self.tree[v]
        queue += ws
        w = queue.pop(0)
        return self.recurse(done, queue, w)

    def find_edge(self, edge):#g.neighbors(u)
        u, v = edge
        u_ground_set = self.recurse([], [], u)
        v_ground_set = self.recurse([], [], v)

        connecting_edges = []
        for u in u_ground_set:
            for ui in self.fully_connected_graph.neighbors(u):
                if ui in v_ground_set:
                    connecting_edges.append( (u, ui) )
        return connecting_edges[np.random.randint(len(connecting_edges))]


    def sample(self):
        nodes = self.all_nodes.copy()
        edges = self.all_edges.copy()

        self.tree = {}
        edges_contracted = []

        def make_g(nds, edg):
            g = nx.Graph()
            g.add_nodes_from(nds)
            g.add_edges_from(edg)
            return g

        g = make_g(nodes, edges)

        for i in tqdm(range(self.n-1, 1, -1)):
            pmf = np.array([resistance_distance(g, e[0], e[1])/i for e in edges])
            ei  = np.random.choice(range(len(edges)), p=pmf/np.sum(pmf))
            
            u, v = edges[ei]
            w = self.contract_edge(u, v)

            nodes = [n for n in nodes if n not in [u, v]] + [w]
            edges_contracted.append(self.find_edge(edges.pop(ei)))

            # CONTRACT u, v BY REPLACING ALL v with w's in edges
            temp = []
            for (ui, vi) in edges:
                t  = [w] if ui in [u, v] else [ui]
                t += [w] if vi in [u, v] else [vi]
                if tuple(t) != (w, w): temp.append(tuple(t))

            edges = temp

            g = make_g(nodes, edges)

        sampled_edges = edges_contracted+[self.find_edge(edges[0])]
        return make_g(self.all_nodes, sampled_edges)

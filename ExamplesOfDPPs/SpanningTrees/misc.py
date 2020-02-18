import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def makeGridGraph(n):
    xs = [i for i in range(n) for _ in range(n)]
    ys = [i for _ in range(n) for i in range(n)]

    pos = {i:(xs[i], ys[i]) for i in range(n*n)} 
    g = nx.Graph()
    g.add_nodes_from(pos.keys())
    for n, p in pos.items():
        g.node[n]['pos'] = p

    edges = []
    options = lambda x,y:[(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    for n, p in pos.items():
        x, y = p
        ops = [(x_, y_) for x_, y_ in options(x, y) if not (x_<0 or x_>=n or y_<0 or y_>=n)]
        matches = [list(pos.keys())[list(pos.values()).index(p_)] for p_ in ops if p_ in list(pos.values())]
        edges += [(n, m) for m in matches]
    seen = set()
    edg = []
    for x in edges:
        if frozenset(x) not in seen:
            edg.append(x)
            seen.add(frozenset(x))
    g.add_edges_from(list(edg))
    return g, list(pos.values())

def plotter(g, pos, title="", nodes_on=True):
    plt.figure(figsize=(6, 6), dpi=300)
    if nodes_on:
        nx.draw_networkx(g, pos=pos, node_color='yellow', with_labels=True, width=2, font_size=7, node_size=80)
    else:
        nx.draw_networkx(g, pos=pos,with_labels=False, node_size=0)
    plt.axis('off')
    plt.title(title)
    plt.savefig("plots/"+title)
    plt.show()
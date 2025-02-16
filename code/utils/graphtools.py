from heapq import heappush, heappop
from itertools import count
import os
import networkx as nx
os.environ['NX_CURGAPH_AUTOCONFIG'] = 'True'

import networkx as nx
import copy as cp



from itertools import islice
def k_shortest_paths(G, source, target, k, weight=None):
    try:
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )
    except nx.NetworkXNoPath:
        return []

# if __name__ == '__main__':
#     G = nx.complete_graph(5)
#     print(k_shortest_paths(G, 0, 4, 4))

from heapq import heappush, heappop
from itertools import count
import os
import networkx as nx
os.environ['NX_CURGAPH_AUTOCONFIG'] = 'True'

# def k_shortest_paths(G, source, target, k=1, weight='weight'):
#     """Returns the k-shortest paths from source to target in a weighted graph G.
#
#     Parameters
#     ----------
#     G : NetworkX graph
#
#     source : node
#        Starting node
#
#     target : node
#        Ending node
#
#     k : integer, optional (default=1)
#         The number of shortest paths to find
#
#     weight: string, optional (default='weight')
#        Edge data key corresponding to the edge weight
#
#     Returns
#     -------
#     lengths, paths : lists
#        Returns a tuple with two lists.
#        The first list stores the length of each k-shortest path.
#        The second list stores each k-shortest path.
#
#     Raises
#     ------
#     NetworkXNoPath
#        If no path exists between source and target.
#
#     Examples
#     --------
#     # >>> G=nx.complete_graph(5)
#     # >>> print(k_shortest_paths(G, 0, 4, 4))
#     ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])
#
#     Notes
#     ------
#     Edge weight attributes must be numerical and non-negative.
#     Distances are calculated as sums of weighted edges traversed.
#
#     """
#     if source == target:
#         return ([0], [[source]])
#
#     length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
#     # print(s)
#     print(length, path)
#     if target not in path:
#         raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
#
#     lengths = [length]
#     paths = [path]
#     c = count()
#     B = []
#     G_original = G.copy()
#
#     for i in range(1, k):
#         for j in range(len(paths[-1]) - 1):
#             spur_node = paths[-1][j]
#             root_path = paths[-1][:j + 1]
#
#             edges_removed = []
#             for c_path in paths:
#                 if len(c_path) > j and root_path == c_path[:j + 1]:
#                     u = c_path[j]
#                     v = c_path[j + 1]
#                     if G.has_edge(u, v):
#                         edge_attr = 1
#                         G.remove_edge(u, v)
#                         edges_removed.append((u, v, edge_attr))
#
#             for n in range(len(root_path) - 1):
#                 node = root_path[n]
#                 # out-edges
#                 for u, v, edge_attr in G.edges_iter(node, data=True):
#                     G.remove_edge(u, v)
#                     edges_removed.append((u, v, edge_attr))
#
#                 if G.is_directed():
#                     # in-edges
#                     for u, v, edge_attr in G.in_edges_iter(node, data=True):
#                         G.remove_edge(u, v)
#                         edges_removed.append((u, v, edge_attr))
#
#             spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)
#             if target in spur_path:
#                 total_path = root_path[:-1] + spur_path
#                 total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length
#                 heappush(B, (total_path_length, next(c), total_path))
#
#             for e in edges_removed:
#                 u, v, edge_attr = e
#                 G.add_edge(u, v)
#
#         if B:
#             (l, _, p) = heappop(B)
#             lengths.append(l)
#             paths.append(p)
#         else:
#             break
#
#     return lengths, paths
#
#
# def get_path_length(G, path, weight='weight'):
#     length = 0
#     if len(path) > 1:
#         for i in range(len(path) - 1):
#             u = path[i]
#             v = path[i + 1]
#
#             length += G.edge[u][v].get(weight, 1)
#
#     return length


import networkx as nx
import copy as cp


# def k_shortest_paths(G, source, target, k=1, weight='weight'):
#     # G is a networkx graph.
#     # source and target are the labels for the source and target of the path.
#     # k is the amount of desired paths.
#     # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.
#     try:
#         length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
#     except nx.NetworkXNoPath:
#         return []
#     A = [path]
#     # print(A)
#     A_len = [length]
#     B = []
#     Gcopy = G
#
#     for i in range(1, k):
#         for j in range(0, len(A[-1]) - 1):
#             edges_removed, node_removed = [], []
#             spurnode = A[-1][j]
#             rootpath = A[-1][:j + 1]
#             for path in A:
#                 if rootpath == path[0:j + 1]:  # and len(path) > j?
#                     if Gcopy.has_edge(path[j], path[j + 1]):
#                         Gcopy.remove_edge(path[j], path[j + 1])
#                         edges_removed.append((path[j], path[j + 1]))
#                     if Gcopy.has_edge(path[j + 1], path[j]):
#                         Gcopy.remove_edge(path[j + 1], path[j])
#                         edges_removed.append((path[j + 1], path[j]))
#             for n in rootpath:
#                 if n != spurnode:
#                     for s in Gcopy.neighbors(n):
#                         edges_removed.append((n, s))
#                         node_removed.append(n)
#                     Gcopy.remove_node(n)
#             try:
#                 _, spurpath = nx.single_source_dijkstra(Gcopy, spurnode, target)
#                 # print(spurpath)
#                 totalpath = rootpath + spurpath[1:]
#                 if totalpath not in B:
#                     B += [totalpath]
#                 for n in node_removed:
#                     Gcopy.add_node(n)
#                 for edge in edges_removed:
#                     Gcopy.add_edge(*edge)
#             except nx.NetworkXNoPath:
#                 for n in node_removed:
#                     Gcopy.add_node(n)
#                 for edge in edges_removed:
#                     Gcopy.add_edge(*edge)
#                 continue
#
#         if len(B) == 0:
#             break
#         lenB = [len(path) for path in B]
#         B = [p for _, p in sorted(zip(lenB, B))]
#         A.append(B[0])
#         A_len.append(sorted(lenB)[0])
#         B.remove(B[0])
#
#     return A


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
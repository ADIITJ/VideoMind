import networkx as nx

# Build temporal graph
def build_temporal_graph(nodes: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for i, (time, caption) in enumerate(nodes):
        G.add_node(i, time=time, caption=caption)
        if i > 0:
            G.add_edge(i-1, i, type='temporal')
    return G

# Gather context window
def gather_context(G: nx.DiGraph, idx: int, before: int = 3, after: int = 3) -> list:
    N = G.number_of_nodes()
    idxs = list(range(max(0, idx-before), min(N, idx+after+1)))
    return [(i, G.nodes[i]['time'], G.nodes[i]['caption']) for i in idxs]
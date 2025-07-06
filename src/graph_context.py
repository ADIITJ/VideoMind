import networkx as nx

# Build temporal graph
def build_temporal_graph(nodes: list) -> nx.DiGraph:
    print("Building temporal graph...")
    G = nx.DiGraph()
    for i, (time, caption) in enumerate(nodes):
        G.add_node(i, time=time, caption=caption)
        if i > 0:
            G.add_edge(i-1, i, type='temporal')
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# Gather context window
def gather_context(G: nx.DiGraph, idx: int, before: int = 3, after: int = 3) -> list:
    print(f"Gathering context for idx={idx}, before={before}, after={after}")
    N = G.number_of_nodes()
    idxs = list(range(max(0, idx-before), min(N, idx+after+1)))
    print(f"Context indices: {idxs}")
    return [(i, G.nodes[i]['time'], G.nodes[i]['caption']) for i in idxs]
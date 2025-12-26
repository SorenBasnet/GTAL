import networkx as nx 
import random 

#
# Generate the Barabasi-Albert graph
#

n = 1000 # Number of nodes
m = 3

# BA graphs are undirected by default
G_undirected = nx.barabasi_albert_graph(n, m)

# Convert to directed graph 
G = G_undirected.to_directed()

# Assign weighted attributes 
# Lambda (λ): Base rate
# Sigma (σ): Noise level

for (u, v) in G.edges():
    G.edges[u, v]['lambda'] = round(random.uniform(0.1, 5.0), 3)
    G.edges[u, v]['sigma'] = round(random.uniform(0.01, 1.0), 3)

# Store as an object/dictionary structure
# Can use adjacency data which includes all attributes

graph_storage = nx.to_dict_of_dicts(G)

sample_edge = list(G.edges())[0]

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Sample edge data {sample_edge}: {graph_storage[sample_edge[0]][sample_edge[1]]}")


"""

“By assigning a baseline rate λᵢⱼ and observation noise σᵢⱼ 
to each edge, the network is parameterized as a stochastic 
generative model for transactional flows. This formulation 
supports discrete-time probabilistic simulation of flow 
propagation under uncertainty and provides a natural 
foundation for later extensions to continuous-time or 
event-driven models.”

"""

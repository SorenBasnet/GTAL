import networkx as nx 
import random 
import numpy as np
import matplotlib.pyplot as plt

#
# Generate the Barabasi-Albert graph
#

# add timesteps to make the Graph Dynamic 
T = 50 
p_missing = 0.05 # Missing data probability 

n = 1000 # Number of nodes
m = 3

true_flows = {} # actual flow : true_flow[t][(u,v)] = f_ij(t)
observed_flows = {} # observed flow : observed_flow[t][(u,v)] = y_ij(t) or None

# BA graphs are undirected by default 
G_undirected = nx.barabasi_albert_graph(n, m) 

# Convert to directed graph 
G = G_undirected.to_directed()

# Assign weighted attributes 
# # Lambda (λ): Base rate 
# # Sigma (σ): Noise level 
for (u, v) in G.edges(): 
    G.edges[u, v]['lambda'] = round(random.uniform(0.1, 5.0), 3) 
    G.edges[u, v]['sigma'] = round(random.uniform(0.01, 1.0), 3)

for t in range(T):
    true_flows[t] = {}
    observed_flows[t] = {}
    
    for (u, v) in G.edges():
        lam = G.edges[u, v]['lambda']
        sigma = G.edges[u, v]['sigma']
        
        # Latent true flow
        f_ij = np.random.poisson(lam)
        
        # Observation noise
        noise = np.random.normal(0, sigma)
        y_ij = f_ij + noise
        
        # Missingness
        if random.random() < p_missing:
            observed_flows[t][(u, v)] = None
        else:
            observed_flows[t][(u, v)] = y_ij
        
        true_flows[t][(u, v)] = f_ij

"""

“By assigning a baseline rate λᵢⱼ and observation noise σᵢⱼ 
to each edge, the network is parameterized as a stochastic 
generative model for transactional flows. This formulation 
supports discrete-time probabilistic simulation of flow 
propagation under uncertainty and provides a natural 
foundation for later extensions to continuous-time or 
event-driven models.”

"""


# check 
t0 = 0
sample_edge = list(G.edges())[0]

print("Sample edge:", sample_edge)
print("True flow:", true_flows[t0][sample_edge])
print("Observed flow:", observed_flows[t0][sample_edge])


# Visualization using matplot lib

edge = sample_edge
true_series = [true_flows[t][edge] for t in range(T)]
obs_series = [observed_flows[t][edge] for t in range(T)]

plt.plot(true_series, label="True Flow")
plt.plot(obs_series, label="Observed Flow", linestyle='--')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Flow")
plt.title("Edge Flow Over Time")
plt.show()


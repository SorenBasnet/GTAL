import networkx as nx 
import random 
import numpy as np
from latent_node_state import Latent_Node_States

class BA_Model: 

    def __init__(self, T, p_missing, n, m):
        self.T = T
        self.p_missing = p_missing
        self.n = n 
        self.m = m 

    def build(self):
        true_flows = {}
        observed_flows = {}

        G_undirected = nx.barabasi_albert_graph(self.n, self.m)
        G = G_undirected.to_directed()

        for (u, v) in G.edges():
            G.edges[u, v]['lambda'] = random.uniform(0.1, 5.0)
            G.edges[u, v]['sigma'] = random.uniform(0.01, 1.0)

        latent_states = Latent_Node_States(G.nodes())

        for t in range(self.T):
            latent_states.step()

            true_flows[t] = {}
            observed_flows[t] = {}

            for (u, v) in G.edges():
                lam_base = G.edges[u, v]['lambda']
                sigma = G.edges[u, v]['sigma']

                mult = latent_states.lambda_multiplier(u, v)
                lam_t = max(lam_base * mult, 1e-6)

                if random.random() < 0.2:
                    f_ij = 0
                else:
                    f_ij = np.random.poisson(lam_t)

                y_ij = max(f_ij + np.random.normal(0, sigma), 0.0)

                true_flows[t][(u, v)] = f_ij

                observed_flows[t][(u, v)] = (
                    None if random.random() < self.p_missing else y_ij
                )

        return G, true_flows, observed_flows

                

              
                
                




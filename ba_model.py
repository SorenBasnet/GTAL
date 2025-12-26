import networkx as nx 
import random 
import numpy as np

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

        # conver to directed 
        G = G_undirected.to_directed()

        # Assign weighted attributes 
        # Lambda (λ): Base rate 
        # Sigma (σ): Noise level 

        for (u,v) in G.edges(): 
            G.edges[u,v]['lambda'] = round(random.uniform(0.1, 5.0), 3)
            G.edges[u,v]['sigma'] = round(random.uniform(0.01, 1.0), 3)

        for t in range(self.T):
            true_flows[t] = {}
            observed_flows[t] = {}

            for (u,v) in G.edges(): 
                lam = G.edges[u, v]['lambda']
                sigma = G.in_edges[u, v]['sigma']

                # Latent true flow 
                f_ij = np.random.poisson(lam)

                #Observation noise 
                noise = np.random.normal(0, sigma)
                y_ij = f_ij + noise 

                # Missingness 

                if random.random() < self.p_missing: 
                    observed_flows[t][(u,v)] = None
                else: 
                    observed_flows[t][(u, v)] = y_ij

                true_flows[t][(u,v)] = f_ij



"""

“By assigning a baseline rate λᵢⱼ and observation noise σᵢⱼ 
to each edge, the network is parameterized as a stochastic 
generative model for transactional flows. This formulation 
supports discrete-time probabilistic simulation of flow 
propagation under uncertainty and provides a natural 
foundation for later extensions to continuous-time or 
event-driven models.”

"""







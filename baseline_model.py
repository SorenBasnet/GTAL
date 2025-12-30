"""
Concept : From the edge (u,v), we try to estimate 
the mean flow, or standard deviation from the 
observed ( non missing) data
"""

import numpy as np 
from collections import defaultdict


class BaselineModel: 

    def __init__(self): 
        self.mean = {}
        self.sigma = {}

    def fit(self, observed_flows, t_range=None):
        """
        Fit per-edge mean and std using only time steps in t_range.
        If t_range is None, uses all time steps.
        """
        if t_range is None:
            t_range = observed_flows.keys()

        edge_values = defaultdict(list)

        for t in t_range:
            for e, v in observed_flows[t].items():
                if v is not None:
                    edge_values[e].append(v)

        for e, vals in edge_values.items():
            self.mean[e] = np.mean(vals)
            self.sigma[e] = np.std(vals) + 1e-6  # avoid division by zero

    def score(self, e, x): 

        """
        z-score of observation x on edge e 
        """

        if e not in self.mean: 
            return None
        
        return (x- self.mean[e])/self.sigma[e]
    
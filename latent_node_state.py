import random

class Latent_Node_States: 

    def __init__(self, nodes, p_switch=0.01): 
        self.states = {i: 0 for i in nodes} # 0 = normal, 1=active
        self.p_switch = p_switch

    def step(self): 
        for i in self.states: 
            if random.random() < self.p_switch: 
                self.states[i] = 1 - self.states[i]

    def lambda_multiplier(self, u, v): 
        if self.states[u] == 1 and self.states[v] == 1: 
            return 2.5
        elif self.states[u] == 1 or self.states[v] == 1: 
            return 1.5
        else: 
            return 1.0
        
    

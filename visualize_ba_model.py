import matplotlib.pyplot as plt

class Visualize_BA_Model: 

    def __init__(self, sample_edge, true_flows, observation_flows, T):
        self.edge = sample_edge 
        self.true_flows = true_flows
        self.observation_flows = observation_flows
        self.T = T


    def sketch_graph(self): 

        true_series = [self.true_flows[t][self.edge] for t in range(self.T)]
        observational_series = [self.observation_flows[t][self.edge] for t in range(self.T)]

        plt.plot(true_series, label="True Flow")
        plt.plot(observational_series, label="Observed Flow", linestyle='--')

        plt.legend()

        plt.xlabel("Time")
        plt.ylabel("Flow")
        plt.title("Edge Flow Over Time")
        plt.show()
        



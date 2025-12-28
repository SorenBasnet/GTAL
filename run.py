from ba_model import BA_Model
from latent_node_state import Latent_Node_States
from visualize_ba_model import Visualize_BA_Model

import matplotlib.pyplot as plt 

if __name__ == '__main__':
    
    model = BA_Model(T=20, p_missing=0.1, n=50, m=2)
    G, true_flows, observed_flows = model.build()

    # Inspect Basic Structure 
    print("Number of nodes : ", G.number_of_nodes())
    print("Number of edges : ", G.number_of_edges())

    print("Time steps : ", len(true_flows))
    print("Edges at t=0 : ", len(true_flows[0]))

    # Visualize the edges 
    edge = list(G.edges())[0]
    viz = Visualize_BA_Model(edge, true_flows, observed_flows, T = 20)
    viz.sketch_graph()

    # Check missingness rate  
    total = 0 
    missing = 0 

    for t in observed_flows: 
        for v in observed_flows[t].values(): 
            total+= 1
            if v is None: 
                missing += 1
    print("Missing rate : ", missing/total)


    # Inspect distributions 

    all_true = []
    all_observation = []

    for t in true_flows: 
        for e in true_flows[t]: 
            all_true.append(true_flows[t][e])
            if observed_flows[t][e] is not None: 
                all_observation.append(observed_flows[t][e])

    plt.hist(all_true, bins=50, alpha=0.6, label="True")
    plt.hist(all_observation, bins=50, alpha=0.6, label="Observed")
    plt.legend()
    plt.title("Flow Distributions")
    plt.show()


    t = 0
    for i, ((u,v), val) in enumerate(observed_flows[t].items()):
        if i >= 5:
            break
        print(f"Edge {(u,v)} | true={true_flows[t][(u,v)]} | obs={val}")
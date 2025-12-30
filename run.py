from ba_model import BA_Model
from latent_node_state import Latent_Node_States
from visualize_ba_model import Visualize_BA_Model
from baseline_model import BaselineModel

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    """
    STEP 0: Freeze parameters (Phase-0 configuration)
    """

    threshold = 3.0

    model = BA_Model(
        T=100,
        p_missing=0.1,
        n=1000,
        m=3
    )

    G, true_flows, observed_flows = model.build()

    anomaly_start = int(0.6 * model.T)
    anomaly_end   = int(0.8 * model.T)

    # Basic sanity checks

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("Time steps:", len(true_flows))
    print("Edges at t=0:", len(true_flows[0]))

    # Visualize a single edge

    edge = list(G.edges())[0]
    viz = Visualize_BA_Model(edge, true_flows, observed_flows, T=20)
    viz.sketch_graph()

    # Missingness check

    total = 0
    missing = 0

    for t in observed_flows:
        for v in observed_flows[t].values():
            total += 1
            if v is None:
                missing += 1

    print("Missing rate:", missing / total)

    # Distribution sanity check

    all_true = []
    all_obs = []

    for t in true_flows:
        for e in true_flows[t]:
            all_true.append(true_flows[t][e])
            if observed_flows[t][e] is not None:
                all_obs.append(observed_flows[t][e])

    plt.hist(all_true, bins=50, alpha=0.6, label="True")
    plt.hist(all_obs, bins=50, alpha=0.6, label="Observed")
    plt.legend()
    plt.title("Flow Distributions")
    plt.show()

    # Baseline learning (PRE-anomaly only)

    baseline = BaselineModel()
    baseline.fit(observed_flows, t_range=range(anomaly_start))

    # STEP 5: Global detection signal

    scores_over_time = []

    for t in range(model.T):
        scores = []
        for e, v in observed_flows[t].items():
            if v is None:
                continue
            scores.append(abs(baseline.score(e, v)))
        scores_over_time.append(np.mean(scores) if scores else 0.0)

    # Detection metrics

    first_detection = None

    for t, score in enumerate(scores_over_time):
        if score > threshold:
            first_detection = t
            break

    if first_detection is None:
        print("No detection occurred.")
    else:
        print("First detection time:", first_detection)
        print("Detection delay:", first_detection - anomaly_start)

    false_alarms = sum(
        1 for t in range(anomaly_start)
        if scores_over_time[t] > threshold
    )

    print("False alarm rate:", false_alarms / anomaly_start)

    # Visualize detection signal

    plt.plot(scores_over_time, label="Mean anomaly score")
    plt.axvspan(anomaly_start, anomaly_end, alpha=0.2, color="red")
    plt.axhline(threshold, linestyle="--", color="black", label="Threshold")
    plt.legend()
    plt.title("Global Detection Signal")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.show()

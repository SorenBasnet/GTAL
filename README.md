# GTAL — Graph Tracking Architecture Learning

[![Python Version](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

**Project Goal**  
Design and validate a probabilistic, uncertainty-aware, temporal graph tracking architecture capable of learning normal transactional network behavior at scale and detecting statistically significant deviations under noisy and incomplete observations.

---

## Phase 0 Technical Objectives

### Objective 1 — Synthetic Transactional World Generator
Create a scalable simulation environment that produces large, multi-layer transactional graphs with realistic statistical behavior and controllable anomalies.

**Deliverables**
- Graph generator (1k–10k nodes)  
- Time-evolving edge flows with delays  
- Noise and missing data models  
- Ground-truth anomaly injection framework  

---

### Objective 2 — Probabilistic Tracking Layer
Develop statistical inference methods to estimate latent network states and predict expected flow behavior under uncertainty.

**Approach**
- Bayesian Networks / Factor Graphs  
- Hidden Markov Models for latent node states  
- Kalman-like or particle filtering adaptations for graphs  

**Outputs**
- Predicted vs. observed flows  
- Posterior belief distributions  
- Missing-data handling  

---

### Objective 3 — Change Detection & Anomaly Reasoning
Formally detect when and where the network deviates from baseline behavior.

**Approach**
- CUSUM / GLR baselines  
- Bayesian Online Change Point Detection  
- Graph-aware deviation metrics  

**Outputs**
- Detection timestamps  
- Spatial localization (which nodes/edges)  
- Confidence scoring  

---

### Objective 4 — AI-Enabled Temporal Graph Learning Layer
Enhance detection performance using learning-based methods beyond hand-crafted statistics.

**Approach**
- Temporal Graph Networks (TGN)  
- Dynamic Graph Neural Networks (DGNN)  
- Comparison against classical baselines  

---

### Objective 5 — Uncertainty Quantification
Quantify both epistemic and aleatoric uncertainty for all detections — a critical metric for AFRL evaluation.

**Metrics**
- Posterior variance  
- Confidence bounds  
- Data sparsity uncertainty  

---

### Objective 6 — Performance Evaluation & Reporting
Demonstrate the architecture’s capability rigorously with measurable outcomes.

**Outputs**
- ROC / PR curves  
- Detection delay distribution  
- False alarm rate control  
- Sensitivity under noise and missing data  
- Scaling performance

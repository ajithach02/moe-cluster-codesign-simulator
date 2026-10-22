# Distributed Mixture-of-Experts (MoE) Cluster Co-Design Simulator

An analytical simulation engine built to model hardware and software co-design bottlenecks during massive-scale MoE model inference (e.g., DeepSeek-V3). 

This simulator quantifies the systemic performance impacts of workload traffic imbalances and network communication bubbles across single-node vs. multi-node GPU cluster topologies under variable user request profiles.

### 🛠️ Co-Design Optimizations Implemented

* **Expert Parallelism Load Balancer (EPLB):** The engine tracks traffic distributions, automatically replicates the weights of the top bottlenecked experts into pre-allocated redundant GPU memory slots, and executes a dispatch-time balancing algorithm to distribute current token queues evenly among replicas.
* **Two-Batch Overlap Pipelining:** The scheduler divides the global token stream into interleaved micro-batches. It pipes the execution timeline so that the network communication phase of Micro-batch B overlaps concurrently with the GPU GEMM math execution of Micro-batch A, completely hiding data transfer latency behind active math operations.

### 📊 Performance Profile Results

The simulation engine sweeps across workload skews under two distinct physical layouts utilizing 4096-token processing stacks:

#### Scenario 1: Single-Node Topology (8x GPUs via 900 GB/s NVLink)
* High intra-node bandwidth mitigates the relative cost of network delays, making math load-balancing the primary bottleneck.
* **Balanced Workload (α=1.01):** 1.65x Processing Speedup
* **Severe Traffic Skew (α=1.50):** 1.31x Processing Speedup

#### Scenario 2: Multi-Node Rack Topology (64x GPUs via 50 GB/s InfiniBand Cables)
* Cross-box token transfers introduce heavy communication bottlenecks, where standard execution routes experience massive delays.
* **Balanced Workload (α=1.01):** **3.50x Acceleration** (Communication latency successfully hidden)
* **Severe Traffic Skew (α=1.50):** 1.35x Processing Speedup 

The dual-topology performance curves are saved visually as `moe_cluster_performance_profile.png`.

---

### 🚀 Getting Started

Ensure you have Python 3.x and dependencies installed:

```bash
pip install numpy matplotlib
python3 main.py
```

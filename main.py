# main.py
import numpy as np
import matplotlib.pyplot as plt
from cluster import MoECluster, GPUSpec, InterconnectSpec
from router import MoERouter
from scheduler import MoESimulator

def run_simulation_for_topology(num_nodes, alphas, h100_spec, network):
    """Runs a sweep over skews for a specific cluster size."""
    cluster = MoECluster(num_nodes=num_nodes, gpus_per_node=8, gpu_spec=h100_spec, interconnect=network)
    naive_latencies = []
    optimized_latencies = []
    speedup_factors = []

    for alpha in alphas:
        router = MoERouter(num_experts=256, top_k=2, zipf_alpha=alpha)
        tokens = router.generate_token_workload(batch_size=4096)

        sim = MoESimulator(cluster)
        naive_res = sim.simulate_naive_expert_parallelism(tokens, token_flops=8e12, token_size_gb=0.15)
        
        sim.implement_eplb_replication(tokens, redundant_slots_per_gpu=2)
        opt_res = sim.simulate_optimized_moe_pipeline(tokens, token_flops=8e12, token_size_gb=0.15)

        naive_ms = naive_res["total_latency_ms"]
        opt_ms = opt_res["total_latency_ms"]
        
        naive_latencies.append(naive_ms)
        optimized_latencies.append(opt_ms)
        speedup_factors.append(naive_ms / opt_ms)
        
    return naive_latencies, optimized_latencies, speedup_factors

def main():
    h100_spec = GPUSpec(name="H100-SXM", hbm_capacity_gb=80, hbm_bandwidth_tbs=3.3, fp8_flops=2e15)
    network = InterconnectSpec(intra_node_bandwidth_gbs=900, inter_node_bandwidth_gbs=50) # Slow cross-box cable
    alphas = np.linspace(1.01, 1.5, 10)

    # 1. RUN SCENARIO A: Single Server Box (8 GPUs total)
    print("🚀 Scenario 1: Single-Node Cluster (8 GPUs over 900 GB/s NVLink)")
    print(f"{'Zipf Alpha (Skew)':<20}{'Naive Latency':<20}{'Optimized Latency':<25}{'Speedup Factor'}")
    print("-" * 75)
    n1_naive, n1_opt, n1_factors = run_simulation_for_topology(1, alphas, h100_spec, network)
    for i, a in enumerate(alphas):
        print(f"{a:<20.2f}{n1_naive[i]:<20.2f}{n1_opt[i]:<25.2f}{n1_factors[i]:.2f}x")

    print("\n" + "="*75 + "\n")

    # 2. RUN SCENARIO B: Massive Data Center Rack (64 GPUs split across 8 boxes)
    print("🚀 Scenario 2: Multi-Node Cluster (64 GPUs over 50 GB/s InfiniBand Cables)")
    print(f"{'Zipf Alpha (Skew)':<20}{'Naive Latency':<20}{'Optimized Latency':<25}{'Speedup Factor'}")
    print("-" * 75)
    n8_naive, n8_opt, n8_factors = run_simulation_for_topology(8, alphas, h100_spec, network)
    for i, a in enumerate(alphas):
        print(f"{a:<20.2f}{n8_naive[i]:<20.2f}{n8_opt[i]:<25.2f}{n8_factors[i]:.2f}x")

    # 3. GENERATE DUAL-TOPOLOGY CHART
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Single Node
    ax1.plot(alphas, n1_naive, color='#e74c3c', linestyle='--', marker='o', label='Naive EP')
    ax1.plot(alphas, n1_opt, color='#e74c3c', linestyle='-', marker='s', label='Optimized Pipeline')
    ax1.set_xlabel('Traffic Skew (Zipf Alpha)', fontweight='bold')
    ax1.set_ylabel('Latency (ms)', color='#e74c3c', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(alphas, n1_factors, color='#2ce3bc', linestyle='-', marker='^', label='Speedup')
    ax1_twin.set_ylabel('Speedup Factor (X)', color='#2ce3bc', fontweight='bold')
    ax1.set_title('Single-Node Architecture\n(Fast NVLink Connections)', fontsize=10, fontweight='bold')

    # Plot 2: Multi Node
    ax2.plot(alphas, n8_naive, color='#e74c3c', linestyle='--', marker='o', label='Naive EP')
    ax2.plot(alphas, n8_opt, color='#e74c3c', linestyle='-', marker='s', label='Optimized Pipeline')
    ax2.set_xlabel('Traffic Skew (Zipf Alpha)', fontweight='bold')
    ax2.set_ylabel('Latency (ms)', color='#e74c3c', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(alphas, n8_factors, color='#2ce3bc', linestyle='-', marker='^', label='Speedup')
    ax2_twin.set_ylabel('Speedup Factor (X)', color='#2ce3bc', fontweight='bold')
    ax2.set_title('Multi-Node Architecture\n(Slower Cross-Box Network Cables)', fontsize=10, fontweight='bold')

    plt.suptitle('Distributed MoE Cluster Scalability Profile:\nQuantifying Workload Skew vs. Network Latency Topology', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('moe_cluster_performance_profile.png', dpi=300)
    print("\n📊 Success! Double-topology benchmarking curves saved as 'moe_cluster_performance_profile.png'")

if __name__ == "__main__":
    main()

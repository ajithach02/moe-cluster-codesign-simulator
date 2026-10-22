import numpy as np
from cluster import MoECluster

class MoESimulator:
    def __init__(self, cluster: MoECluster, num_experts: int = 256):
        self.cluster = cluster
        self.num_experts = num_experts
        self.expert_mapping = {}
        for exp_id in range(num_experts):
            self.expert_mapping[exp_id] = [exp_id % cluster.total_gpus]

    def simulate_naive_expert_parallelism(self, token_assignments: np.ndarray, token_flops: float, token_size_gb: float) -> dict:
        batch_size = token_assignments.shape[0]
        gpu_token_counts = np.zeros(self.cluster.total_gpus)
        gpu_incoming_comm_gb = np.zeros(self.cluster.total_gpus)
        
        for token_idx in range(batch_size):
            for expert_id in token_assignments[token_idx]:
                assigned_gpu = self.expert_mapping[expert_id][0]
                gpu_token_counts[assigned_gpu] += 1
                origin_gpu = token_idx % self.cluster.total_gpus
                if origin_gpu != assigned_gpu:
                    gpu_incoming_comm_gb[assigned_gpu] += token_size_gb

        network_latencies = []
        for gpu_id in range(self.cluster.total_gpus):
            if gpu_incoming_comm_gb[gpu_id] > 0:
                peer_gpu = (gpu_id + 1) % self.cluster.total_gpus
                lat = self.cluster.get_transfer_time(gpu_incoming_comm_gb[gpu_id], peer_gpu, gpu_id)
                network_latencies.append(lat)
        
        comm_latency = max(network_latencies) if network_latencies else 0.0
        compute_latencies = [counts * token_flops / self.cluster.gpu_spec.fp8_flops for counts in gpu_token_counts]
        max_compute_latency = max(compute_latencies)
        total_latency = max_compute_latency + comm_latency
        
        return {
            "total_latency_ms": total_latency * 1000,
            "compute_time_ms": max_compute_latency * 1000,
            "comm_time_ms": comm_latency * 1000,
            "bubble_percentage": (comm_latency / total_latency) * 100 if total_latency > 0 else 0
        }

    def implement_eplb_replication(self, historical_token_counts: np.ndarray, redundant_slots_per_gpu: int = 2):
        expert_loads = np.zeros(self.num_experts)
        for tokens in historical_token_counts:
            for exp_id in tokens:
                expert_loads[exp_id] += 1
                
        hot_expert_ids = np.argsort(expert_loads)[::-1]
        gpu_slots = {gpu_id: redundant_slots_per_gpu for gpu_id in range(self.cluster.total_gpus)}
        
        for exp_id in hot_expert_ids[:self.cluster.total_gpus]:
            sorted_gpus = sorted(gpu_slots.keys(), key=lambda g: gpu_slots[g], reverse=True)
            for target_gpu in sorted_gpus:
                if target_gpu not in self.expert_mapping[exp_id] and gpu_slots[target_gpu] > 0:
                    self.expert_mapping[exp_id].append(target_gpu)
                    gpu_slots[target_gpu] -= 1
                    break 

    def simulate_optimized_moe_pipeline(self, token_assignments: np.ndarray, token_flops: float, token_size_gb: float) -> dict:
        midpoint = len(token_assignments) // 2
        micro_batches = [token_assignments[:midpoint], token_assignments[midpoint:]]
        mb_compute_latencies = []
        mb_comm_latencies = []
        
        for mb in micro_batches:
            gpu_token_counts = np.zeros(self.cluster.total_gpus)
            gpu_comm_gb = np.zeros(self.cluster.total_gpus)
            
            for token in mb:
                for exp_id in token:
                    replicas = self.expert_mapping[exp_id]
                    assigned_gpu = min(replicas, key=lambda g: gpu_token_counts[g])
                    gpu_token_counts[assigned_gpu] += 1
                    origin_gpu = (assigned_gpu + 1) % self.cluster.total_gpus
                    if origin_gpu != assigned_gpu:
                        gpu_comm_gb[assigned_gpu] += token_size_gb
                        
            mb_compute_latencies.append(max(gpu_token_counts * token_flops / self.cluster.gpu_spec.fp8_flops))
            mb_comm_latencies.append(max([self.cluster.get_transfer_time(gb, (g+1)%self.cluster.total_gpus, g) for g, gb in enumerate(gpu_comm_gb)]) if max(gpu_comm_gb) > 0 else 0)

        t0 = mb_comm_latencies[0]
        t1 = max(mb_compute_latencies[0], mb_comm_latencies[1]) 
        t2 = mb_compute_latencies[1]
        
        total_pipelined_latency = t0 + t1 + t2
        bubble_latency = max(0, total_pipelined_latency - sum(mb_compute_latencies))

        return {
            "total_latency_ms": total_pipelined_latency * 1000,
            "bubble_percentage": (bubble_latency / total_pipelined_latency) * 100 if total_pipelined_latency > 0 else 0
        }

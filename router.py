import numpy as np

class MoERouter:
    def __init__(self, num_experts: int, top_k: int, zipf_alpha: float = 1.2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.zipf_alpha = zipf_alpha  

    def generate_token_workload(self, batch_size: int) -> np.ndarray:
        raw_samples = np.random.zipf(self.zipf_alpha, size=(batch_size, self.top_k))
        expert_assignments = (raw_samples - 1) % self.num_experts
        for i in range(batch_size):
            while len(np.unique(expert_assignments[i])) < self.top_k:
                expert_assignments[i] = np.random.choice(self.num_experts, size=self.top_k, replace=False)
        return expert_assignments

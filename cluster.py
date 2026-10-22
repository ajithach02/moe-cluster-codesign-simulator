from dataclasses import dataclass

@dataclass
class GPUSpec:
    name: str
    hbm_capacity_gb: float
    hbm_bandwidth_tbs: float  
    fp8_flops: float          

@dataclass
class InterconnectSpec:
    intra_node_bandwidth_gbs: float  
    inter_node_bandwidth_gbs: float  

class MoECluster:
    def __init__(self, num_nodes: int, gpus_per_node: int, gpu_spec: GPUSpec, interconnect: InterconnectSpec):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = num_nodes * gpus_per_node
        self.gpu_spec = gpu_spec
        self.interconnect = interconnect
        
    def get_transfer_time(self, data_size_gb: float, src_gpu: int, dst_gpu: int) -> float:
        if src_gpu == dst_gpu:
            return 0.0
        src_node = src_gpu // self.gpus_per_node
        dst_node = dst_gpu // self.gpus_per_node
        if src_node == dst_node:
            return data_size_gb / self.interconnect.intra_node_bandwidth_gbs
        return data_size_gb / self.interconnect.inter_node_bandwidth_gbs

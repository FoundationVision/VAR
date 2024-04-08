# main.py
from distributed import DistributedManager, master_only, local_master_only, for_visualize
import torch

class DistributedApp:
    def __init__(self):
        self.manager = DistributedManager()
        self.manager.initialize()

    @master_only
    def main(self):
        rank = self.manager.get_rank()
        world_size = self.manager.get_world_size()
        local_rank = self.manager.get_local_rank()

        print(f"Rank: {rank}, World size: {world_size}, Local Rank: {local_rank}")

        tensor = torch.tensor([rank], dtype=torch.float32)
        tensor = self.manager.allreduce(tensor)
        print(f"Rank: {rank}, Reduced Tensor: {tensor.item()}")

    def finalize(self):
        self.manager.finalize()

if __name__ == "__main__":
    app = DistributedApp()
    app.main()
    app.finalize()

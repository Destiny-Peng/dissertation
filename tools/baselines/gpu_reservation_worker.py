"""Hand LF3R's PyTorch reservation to the vLLM CUDA worker."""
from vllm.v1.worker.gpu_worker import Worker
from gpu_memory_reservation import release_reserved_device


class ReservedGPUWorker(Worker):
    def init_device(self):
        release_reserved_device(self.local_rank)
        return super().init_device()

from typing import Any, Dict, Optional, Callable
import gc
import time

import psutil
import torch

from src.config import ModelConfig
from src.metrics.runtime_cost import CPUMemoryMonitor


class Tool:
    """
    A utility class for executing model inference or other processes
    with optional cost measurement and resource monitoring.
    """
    config: ModelConfig
    model: Optional[torch.nn.Module]
    process: Optional[Callable[..., Any]]
    options: Dict[str, Any]
    _device: str

    def __init__(
            self,
            config: ModelConfig,
            model: torch.nn.Module,
            *,
            process: Optional[Callable[..., Any]] = None,
            device: str = "cpu",
            **kwargs: Any
    ):
        """
        Initialize the Tool with a given model configuration, model instance,
        an optional processing function, and device preference.

        :param config: The model configuration object.
        :param model: The PyTorch model to be used.
        :param process: A callable that executes operations on the model.
        :param device: The device type ('cpu' or 'cuda:x').
        :param kwargs: Any extra options or parameters to store.
        """
        self.config = config
        self.model = model
        self.process = process
        self.options = kwargs

        # Set and move model to the preferred device
        self.device = device
        # Switch the model to evaluation mode by default
        self.model.eval()

    @property
    def device(self) -> str:
        """
        Current device to which the model is allocated.
        """
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        """
        Move the model to the specified device and update the internal record.
        """
        self._device = device
        # Always move model to the newly set device, even if it's CPU.
        self.model.to(device)

    def to(self, device: str) -> None:
        """
        Manually move the model to a specified device.

        :param device: The target device type, e.g., 'cpu' or 'cuda:0'.
        """
        self.device = device  # Reuse the device setter

    def execute(
            self,
            *args: Any,
            cost_aware: bool = False,
            **kwargs: Any
    ) -> Any:
        """
        Execute the provided processing function on the model,
        optionally measuring resource usage (CPU/GPU memory, execution time).

        :param cost_aware: If True, measure resources and return the stats.
        :param args: Positional arguments to pass to the process function.
        :param kwargs: Keyword arguments to pass to the process function.
        :return: If cost_aware is False, returns the result of the process.
                 If cost_aware is True, returns a tuple (result, costs).
        """
        # If not measuring resource usage, run immediately and return
        if not cost_aware:
            return self.process(*args, **kwargs, device=self.device)

        # Resource cleanup before measurement
        if self.device != 'cpu':
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()

        # Prepare monitors and baseline measurements
        cpu_mem_monitor = CPUMemoryMonitor(interval=0.1)
        time_before = time.perf_counter() * 1000
        cpu_mem_before = psutil.Process().memory_info().rss / (1024 ** 2)
        gpu_mem_before = 0
        if self.device != 'cpu':
            gpu_mem_before = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        # Start CPU memory monitoring and run process
        cpu_mem_monitor.start()
        result = self.process(*args, **kwargs, device=self.device)
        if self.device != 'cpu':
            torch.cuda.synchronize()
        cpu_mem_monitor.stop()

        # Get time and memory usage differences
        time_after = time.perf_counter() * 1000
        cpu_mem_after = cpu_mem_monitor.get_max_cpu_memory_allocated(unit='MB')
        gpu_mem_after = 0
        if self.device != 'cpu':
            gpu_mem_after = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        costs = {
            'exec_time': time_after - time_before,
            'short_term_cpu_memory': cpu_mem_after - cpu_mem_before,
            'short_term_gpu_memory': gpu_mem_after - gpu_mem_before
        }

        return result, costs

    def __repr__(self) -> str:
        """
        Return a string representation of the Tool
        by delegating to the underlying config object.
        """
        return repr(self.config)

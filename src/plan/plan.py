import gc
from collections import deque
from typing import Any, Dict, Set, Deque, Tuple

import torch

from src.config import DEFAULT_START_TASK_NAME, TOOL_DEVICE_LIST, log
from src.utils import get_available_device, normalize_task_name
from src.types import TaskName, ModelName, CostInfo
from src.tools.tool import Tool
from src.tools.tool_manager import tool_manager
from .plan_graph import NodeID, PlanNode, PlanGraph


class Plan:
    """
    A Plan object that encapsulates a PlanGraph along with the tools needed to execute it.
    It manages the lifecycle of the tools (prepare & clean) and provides methods to execute
    the plan graph and collect final results.
    """

    graph: PlanGraph
    tools: Dict[Tuple[TaskName, ModelName], Tool]
    is_done: bool
    price: float
    exec_time: float

    def __init__(self, plan_info: Any = None) -> None:
        """
        Initialize the Plan with a PlanGraph and optional plan_info to build the graph structure.

        Args:
            plan_info: A structure describing how to build the plan graph.
        """
        self.graph = PlanGraph()
        self.tools = {}
        self.is_done = False
        self.price = 0.0
        self.exec_time = 0.0

        # If plan_info is provided, build the graph from it.
        if plan_info:
            try:
                self.create_graph_from_plan_info(plan_info)
            except IndexError:
                log.info("Invalid plan detected. Please check the plan structure.")

    def create_graph_from_plan_info(self, plan_info: Any) -> None:
        """
        Create a plan graph from a plan_info object (e.g., a list describing tasks and dependencies).

        The format might be something like:
            [task_name, [dependency_task_name], task_name, [dependency_task_name], ...]
        Example:
            ['Image Deblurring', ['Input query'], 'Image Denoising', ['Image Deblurring'], ...]

        Currently, an assert enforces that each task has at least one dependency.
        """
        for i in range(0, len(plan_info), 2):
            task_name = plan_info[i]
            dependencies = plan_info[i + 1]

            assert (
                    len(dependencies) >= 1
            ), "At least one dependency required per operation."
            task_name = normalize_task_name(task_name)
            target_node = self.graph.add_node(task_name)
            for dependency in dependencies:
                if isinstance(dependency, list):
                    dependency = dependency[0]

                dependency = normalize_task_name(dependency)
                source_node = self.graph.get_or_add_node(dependency)
                self.graph.add_edge(source_node, target_node)

    def prepare_tools(self) -> None:
        """
        Prepare (load and allocate) all tools required by the plan.
        Moves tools to an available device if they are CPU-based by default.
        """
        for node in self.graph.nodes.values():
            task_name = node.task_name
            model_name = node.model_name
            if task_name == DEFAULT_START_TASK_NAME:
                continue

            tool_key = (task_name, model_name)
            if tool_key not in self.tools:
                self.tools[tool_key] = tool_manager.get_model(task_name, model_name)

            tool = self.tools[tool_key]
            # If the tool is CPU-based, we attempt to move it to an available device (GPU) if possible
            if tool.device == "cpu":
                device = get_available_device(TOOL_DEVICE_LIST)
                tool.to(device)

    def clean_tools(self) -> None:
        """
        Release or unload all tools used in the plan.
        Moves them back to 'cpu', clears CUDA caches, and forces garbage collection.
        """
        used_devices = set()

        for tool in self.tools.values():
            if tool.device != "cpu":
                used_devices.add(tool.device)
                tool.to("cpu")

        for device in used_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        gc.collect()

    def _execute_on_graph(self, input_data: Any, *, cost_aware) -> None:
        """
        Execute the plan graph with the given input data.

        Note: Assuming the plan graph is given in the form of a topologically sorted list of nodes and edges,
        then we can execute the plan along the nodes list.

        Why we do this instead of using search algorithms? Just easy for debug, and don't waste a good nature :D

        Args:
            input_data: The initial input to be stored in the start node.
        """

        for node in self.graph.nodes.values():
            if node.is_start_point:
                node.set_value(input_data)
                DEFAULT_COSTS: CostInfo = {
                    "exec_time": 0.0,
                    "short_term_cpu_memory": 0.0,
                    "short_term_gpu_memory": 0.0,
                }
                node.costs = DEFAULT_COSTS
                continue

            # Get the input data from all parent nodes
            current_input = {}
            for edge_ref in node.in_edges.values():
                edge = edge_ref()
                source_node = edge.source()
                current_input.update(source_node.get_value())

            tool = self.tools[(node.task_name, node.model_name)]
            result, costs = tool.execute(current_input, cost_aware=cost_aware)
            node.set_value(result)
            node.costs = costs

    def collect_results(self) -> Dict[TaskName, Any]:
        """
        Collect results from all end-point nodes in the plan graph.
        Returns a dictionary combining the results of all end-point nodes.
        """
        results = {}
        for node in self.graph.nodes.values():
            if node.is_end_point:
                node_value = node.get_value()
                if isinstance(node_value, dict):
                    results.update(node_value)
                else:
                    results[node.task_name] = node_value

        return results

    def calculate_price_and_save(self) -> float:
        """
        Calculate the total price for executing all nodes in the plan,
        based on each node's cost metrics and pricing configuration in PlanNode.
        """
        price_sum = 0.0
        for node in self.graph.nodes.values():
            if node is self.graph.start_node:
                node.price = 0.0
            else:
                node.calculate_price_and_save()

            price_sum += node.price

        self.price = price_sum
        return price_sum

    def calculate_exec_time_and_save(self) -> float:
        """
        Calculate the total (critical path) execution time of the plan using BFS-like traversal.
        This sets each node's 'critical_exec_time' to the sum of its own exec_time and
        the maximum exec_time among its predecessors.
        """
        exec_time_total = 0.0

        visited: Set[NodeID] = set()
        queue: Deque[PlanNode] = deque()

        start_node = self.graph.start_node
        start_node.critical_exec_time = 0.0
        visited.add(start_node.node_id)
        queue.append(start_node)

        while queue:
            current_node = queue.popleft()

            # The maximum of all parent nodes' critical_exec_time
            max_parent_time = 0.0
            for _, edge_ref in current_node.in_edges.items():
                edge = edge_ref()
                source_node = edge.source()
                if source_node and source_node.critical_exec_time is not None:
                    max_parent_time = max(
                        max_parent_time, source_node.critical_exec_time
                    )

            # Add this node's own time
            node_exec_time = current_node.costs["exec_time"] if current_node.costs else 0.0
            current_node.critical_exec_time = node_exec_time + max_parent_time

            exec_time_total = max(exec_time_total, current_node.critical_exec_time)

            for _, edge_ref in current_node.out_edges.items():
                edge = edge_ref()
                target_node = edge.target()
                if target_node.node_id not in visited:
                    visited.add(target_node.node_id)
                    queue.append(target_node)

        self.exec_time = exec_time_total
        return exec_time_total

    def execute(self, input_data: Any, *, cost_aware=True) -> Any:
        """
        Prepare tools, execute the plan, collect results, then clean up.
        Returns the collected results from all end-point nodes.
        """
        self.prepare_tools()
        log.info("Tool preparation done. Executing plan...")
        try:
            self._execute_on_graph(input_data, cost_aware=cost_aware)
            log.info("Plan execution done. Collecting results...")
            results = self.collect_results()
            self.calculate_exec_time_and_save()
            self.calculate_price_and_save()
        except (KeyError, TypeError):
            log.info("Invalid plan detected. Please check the plan structure.")
            results = None
        self.is_done = True
        self.clean_tools()
        log.info("Tools clean up. If you want to clean up the entire plan, call plan.cleanup().")

        return results

    def cleanup(self, clean_tools=True) -> None:
        """
        Clean up the entire Plan, including:
        1. Clear and reset all attributes, including graph, tools, is_done, price, exec_time, etc.
        2. Call clean_tools() to release or unload all tools;
        """
        self.graph = PlanGraph()
        self.tools.clear()
        self.is_done = False
        self.price = 0.0
        self.exec_time = 0.0
        if clean_tools:
            self.clean_tools()

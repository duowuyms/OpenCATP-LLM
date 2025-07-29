import weakref
from bisect import bisect_left
from typing import Any, Dict, Optional

from src.config import DEFAULT_START_TASK_NAME, GlobalMetricsConfig as Mcfg
from src.types import TaskName, ModelName, CostInfo

NodeID = int
EdgeID = int


class PlanGraph:
    """
    A directed graph that maintains strong references to all nodes and edges.
    Provides methods to add or remove nodes and edges.
    """

    name_to_id: Dict[TaskName, NodeID]
    nodes: Dict[NodeID, "PlanNode"]
    edges: Dict[EdgeID, "PlanEdge"]
    _next_node_id: int
    _next_edge_id: int

    def __init__(self) -> None:
        """
        Initialize the graph. Automatically create a default start node
        based on DEFAULT_START_TASK_NAME.
        """
        self._next_node_id = 0
        self._next_edge_id = 0

        # Stores Node/Edge objects by their ID
        self.nodes = {}
        self.edges = {}

        # Maps task_name -> node_id (unique in this graph)
        self.name_to_id = {}

        # Create a default start node
        self.add_node(task_name=DEFAULT_START_TASK_NAME, is_start_point=True)

    @property
    def start_node(self) -> "PlanNode":
        """
        Return the default start node of the graph, which is always node_id=0.
        """
        return self.nodes[0]

    def add_node(
        self,
        task_name: TaskName,
        *,
        model_name: Optional[ModelName] = None,
        is_start_point: bool = False,
        is_end_point: bool = True,
        is_done: bool = False,
        value: Any = None,
    ) -> "PlanNode":
        """
        Create a new node in the graph with the given task_name, and return it.
        Raises ValueError if a node with the same task_name already exists.
        """
        if task_name in self.name_to_id:
            raise ValueError(f"Node with task name '{task_name}' already exists.")

        node_id = self._next_node_id
        self._next_node_id += 1

        node = PlanNode(
            node_id=node_id,
            task_name=task_name,
            model_name=model_name,
            is_start_point=is_start_point,
            is_end_point=is_end_point,
            is_done=is_done,
            value=value,
        )
        # Use a weak reference to avoid circular references
        node.graph = weakref.ref(self)

        # Register this node in the graph's internal dict
        self.nodes[node_id] = node
        self.name_to_id[task_name] = node_id

        return node

    def get_or_add_node(
        self, task_name: TaskName, model_name: ModelName = None
    ) -> "PlanNode":
        """
        Retrieve the node with the given task_name.
        If it does not exist, a new one is created.
        """
        # todo Fix: Now only support a single global tool/model per task name within a plan
        node_id = self.name_to_id.get(task_name)
        if node_id is None:
            return self.add_node(task_name, model_name)
        return self.nodes[node_id]

    def add_edge(self, source: "PlanNode", target: "PlanNode") -> "PlanEdge":
        """
        Create a new directed edge from `source` to `target` and add it to the graph.
        Returns the newly created PlanEdge object.
        """
        edge_id = self._next_edge_id
        self._next_edge_id += 1

        edge = PlanEdge(edge_id=edge_id, source=source, target=target)
        # Use a weak reference to the graph
        edge.graph = weakref.ref(self)

        # Store the edge in the graph
        self.edges[edge_id] = edge

        source.is_end_point = False
        # Both source and target hold weak references to this edge
        source.out_edges[edge_id] = weakref.ref(edge)
        target.in_edges[edge_id] = weakref.ref(edge)

        return edge

    def remove_node(self, node_id: NodeID) -> None:
        """
        Remove the node with the specified node_id from the graph,
        along with any incoming and outgoing edges.
        """
        node = self.nodes.pop(node_id, None)
        # Do not remove the default start node with ID=0
        if not node or node.node_id == 0:
            return

        # If present, remove mapping from task_name to node_id
        if self.name_to_id.get(node.task_name) == node_id:
            self.name_to_id.pop(node.task_name, None)

        # Collect all edges connected to this node to avoid iteration conflicts
        related_edge_ids = set(node.in_edges.keys()) | set(node.out_edges.keys())

        # Remove each related edge
        for e_id in related_edge_ids:
            self.remove_edge(e_id)

    def remove_edge(self, edge_id: EdgeID) -> None:
        """
        Remove the edge with the specified edge_id from the graph.
        Also updates source/target node references.
        """
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return

        source_node = edge.source()
        target_node = edge.target()

        # Clean up references in the source node
        if source_node is not None:
            source_node.out_edges.pop(edge_id, None)

        # Clean up references in the target node
        if target_node is not None:
            target_node.in_edges.pop(edge_id, None)


class PlanNode:
    """
    Represents a node in the PlanGraph.
    Stores node-related attributes such as task_name, model_name, etc.
    """

    node_id: NodeID
    task_name: TaskName
    model_name: Optional[ModelName]
    is_start_point: bool
    is_end_point: bool
    is_done: bool
    value: Any
    costs: Optional[CostInfo]
    price: Optional[float]
    critical_exec_time: Optional[float]

    graph: Optional[weakref.ReferenceType["PlanGraph"]]
    in_edges: Dict[EdgeID, weakref.ReferenceType["PlanEdge"]]
    out_edges: Dict[EdgeID, weakref.ReferenceType["PlanEdge"]]

    def __init__(
        self,
        node_id: NodeID,
        task_name: TaskName,
        model_name: Optional[ModelName] = None,
        is_start_point: bool = False,
        is_end_point: bool = True,
        is_done: bool = False,
        value: Any = None,
    ) -> None:
        self.node_id = node_id
        self.task_name = task_name
        self.model_name = model_name
        self.is_start_point = is_start_point
        self.is_end_point = is_end_point
        self.is_done = is_done
        self.value = value

        self.costs = None
        self.price = None
        self.critical_exec_time = None

        # Will be set by PlanGraph to avoid circular reference
        self.graph = None

        # Weak references to edges
        self.in_edges = {}
        self.out_edges = {}

    def get_value(self) -> Any:
        """
        Get the node's value (if any).
        """
        return self.value

    def set_value(self, value: Any) -> None:
        """
        Assign a value to this node and mark it as done.
        """
        self.value = value
        self.is_done = True

    def calculate_price_and_save(self) -> float:
        """
        Calculate the execution price based on the node's cost metrics
        and the global pricing configuration in Mcfg.

        Price = exec_time * (
            long_term_cpu_mem * long_term_cpu_price_unit
          + short_term_cpu_mem * short_term_cpu_price_unit
          + long_term_gpu_mem * long_term_gpu_price_unit
          + short_term_gpu_mem * short_term_gpu_price_unit
        ) + price_per_request
        """
        if self.costs is None:
            raise RuntimeError("Costs information is not set for this node.")

        # Price for short-term CPU and GPU
        short_term_cpu_price = (
            self.costs["short_term_cpu_memory"] * Mcfg.cpu_short_memory_pricing_per_mb
        )
        short_term_gpu_price = (
            self.costs["short_term_gpu_memory"] * Mcfg.gpu_short_memory_pricing_per_mb
        )

        long_term_cpu_memory = Mcfg.tools_cpu_long_term_mem[self.task_name]
        long_term_gpu_memory = Mcfg.tools_gpu_long_term_mem[self.task_name]

        long_term_cpu_memory_tiers = sorted(Mcfg.cpu_long_memory_pricing.keys())
        long_term_gpu_memory_tiers = sorted(Mcfg.gpu_long_memory_pricing.keys())

        # Find the proper price unit by searching the correct tier
        cpu_index = bisect_left(long_term_cpu_memory_tiers, long_term_cpu_memory)
        cpu_index = min(
            cpu_index, len(long_term_cpu_memory_tiers) - 1
        )  # clamp index if needed
        gpu_index = bisect_left(long_term_gpu_memory_tiers, long_term_gpu_memory)
        gpu_index = min(
            gpu_index, len(long_term_gpu_memory_tiers) - 1
        )  # clamp index if needed

        long_term_cpu_price_unit = Mcfg.cpu_long_memory_pricing[
            long_term_cpu_memory_tiers[cpu_index]
        ]
        long_term_gpu_price_unit = Mcfg.gpu_long_memory_pricing[
            long_term_gpu_memory_tiers[gpu_index]
        ]

        price = (
            self.costs["exec_time"]
            * (
                long_term_cpu_memory * long_term_cpu_price_unit
                + short_term_cpu_price
                + long_term_gpu_memory * long_term_gpu_price_unit
                + short_term_gpu_price
            )
            + Mcfg.price_per_request
        )

        self.price = price
        return price


class PlanEdge:
    """
    Represents an edge in the PlanGraph.
    Holds weak references to its source and target PlanNodes, as well as the PlanGraph.
    """

    edge_id: EdgeID
    graph: Optional[weakref.ReferenceType["PlanGraph"]]
    source: weakref.ReferenceType["PlanNode"]
    target: weakref.ReferenceType["PlanNode"]

    def __init__(
        self,
        edge_id: EdgeID,
        source: "PlanNode",
        target: "PlanNode",
    ) -> None:
        self.edge_id = edge_id

        # The PlanGraph will set this to a weak reference
        self.graph = None

        # Source and target nodes are stored via weak references
        self.source = weakref.ref(source)
        self.target = weakref.ref(target)

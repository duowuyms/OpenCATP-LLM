from typing import cast

import networkx as nx
import matplotlib.pyplot as plt
import torch

from src.config import TOOL_GPU_MEMORY_ALLOC_LIMIT, DEFAULT_START_TASK_NAME
from src.types import TaskName
from src.plan.plan_graph import PlanGraph


def get_available_device(device_list):
    """
    Return the first device in the list that has enough free memory.
    If none, return "cpu".

    Args:
        device_list: List of CUDA devices to choose from.

    Returns:
        str: The device name that has enough free memory or "cpu".
    """

    def qualifies(device):
        free_mem, _ = torch.cuda.mem_get_info(torch.device(device))
        return free_mem > TOOL_GPU_MEMORY_ALLOC_LIMIT

    return next((d for d in device_list if qualifies(d)), "cpu")


def normalize_task_name(source: str) -> TaskName:
    source = source.strip()
    source = source.replace(" ", "_").lower()

    # Normalize special cases
    # fixme: name alias for backward compatibility
    if 'input' in source:
        source = DEFAULT_START_TASK_NAME

    match source:
        case "input_of_query":
            source = DEFAULT_START_TASK_NAME
        case 'input_query':
            source = DEFAULT_START_TASK_NAME
        case 'colorization':
            source = 'image_colorization'
        case 'text_to_image_generation':
            source = 'text_to_image'
        case 'fill_mask':
            source = 'mask_filling'
        case _:
            pass
    target = cast(TaskName, source)
    return target


def print_graph(graph: PlanGraph, *, save_path=None):
    nodes = graph.nodes
    edges = graph.edges

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for edge in edges.values():
        G.add_edge(edge.source().node_id, edge.target().node_id)

    nx.draw(G, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_weight='bold')

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def get_model_parameters_size(model: torch.nn.Module, size: str = 'mb'):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()

    if size == 'kb':
        total_size = total_bytes / 1024
    elif size == 'mb':
        total_size = total_bytes / (1024 ** 2)
    else:
        total_size = total_bytes

    return total_size

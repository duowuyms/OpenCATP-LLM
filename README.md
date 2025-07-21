# OpenCATP-LLM

Hi! This is the official codebase of CATP-LLM and OpenCATP of the paper ``CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning'' (ICCV 2025).

What is *CATP-LLM*?: CATP-LLM is the first framework that enables the LLMs to effectively balance the trade-off between task performance and execution costs (e.g., tool execution time) during tool planning or tool using.

What is *OpenCATP*?: OpenCATP is the first platform to evaluate the performance of LLMs in cost-aware tool planning, which features diverse tools and difficult tasks, as well as a comprehensive scheme to measure the tool execution costs.

TODO List:

- [x] Release OpenCATP.
- [x] Release CATP-LLM.
  - [x] Release Training Codes.
  - [x] Release Testing Codes.
- [x] Test codes.
- [ ] Release model weights of CATP-LLM (after acceptance).

We will keep updating this repo following the TODO list. This may take some time. Please stay tuned! :)


## Content

- [Getting Started](#getting-started)
- [Source Code](#source-code)
  - [Part 1: OpenCATP](#part-1-opencatp)
  - [Part 2: CATP-LLM](#part-2-catp-llm)
- [How to add new baselines to this repo?](#how-to-add-new-baselines-to-this-repo)
- [Citation](#citation)
## Getting Started 

1. Install Python (recommended version >= 3.11.9).

2. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. On the first run, tools and HF models will be dynamically downloaded to the `hf_cache` path specified in the config. For tasks using GitHub-based models, please refer to the corresponding subdirectory’s README (or check the runtime error messages) to download the model weights and place them in the specified folder.

4. Please configure the config manually or place the dataset under `./dataset`. The dataset can be obtained from [this link](https://drive.google.com/file/d/1mbrBdA0xu_dzwCzDAIAC1dlyEa5vVKWq/view?usp=drive_link).

5. For a simple test case of running OpenCATP, please run:
    ```sh 
    python test_opencatp.py
    ```
6. For a simple case of running CATP-LLM, please run:
    ```sh 
    python run_catpllm.py --config_file src/catpllm/data/config_data/default_debug.yaml
    ```
7. If you want to run CATP-LLM on sequential or non-sequential tasks in OpenCATP, please run:
    ```sh 
    # sequential tasks
    python run_catpllm.py --config_file src/catpllm/data/config_data/default_seq.yaml
    # non-sequential tasks
    python run_catpllm.py --config_file src/catpllm/data/config_data/default_nonseq.yaml
    ```
    You can customize your own settings by modifying the .yaml file.

**NOTE:** It is recommended to configure the several paths specificied in `src/config.GlobalPathConfig` before running this repo. It is also recommended to use soft links for path configuration.

## Source Code

The main codes are placed in the `src` directory:

### Part 1: OpenCATP

The main codes of OpenCATP are listed below:
```
./src
├── config.py
├── data_loader.py
├── __init__.py
├── metrics
│   ├── evaluator.py
│   ├── __init__.py
│   └── runtime_cost.py
├── plan
│   ├── __init__.py
│   ├── plan_graph.py
│   └── plan.py
├── tools
│   ├── github_models
│   ├── grouped_tools.py
│   ├── __init__.py
│   ├── tool_manager.py
│   └── tool.py
├── types.py
└── utils.py
```

Some key codes are explained as follows.

#### 1. `src/plan/plan.py`

#### Main Class

#### `Plan`

- **Purpose**
   Encapsulates an execution plan (`PlanGraph`) along with all required tools (`Tool`), and manages tool lifecycles. Offers methods to run the entire plan graph, calculate cost and runtime, and collect the final output.
- **Primary Attributes**
  - `graph`: A `PlanGraph` instance representing the directed graph of the entire workflow.
  - `tools`: A dictionary whose keys are `(TaskName, ModelName)` tuples, and values are the corresponding `Tool` instances.
  - `is_done`: A boolean indicating whether the plan has finished execution.
  - `price`: A `float` storing the accumulated cost after executing all nodes.
  - `exec_time`: A `float` indicating the total runtime of the critical path.
- **Key Methods**
  1. `__init__(plan_info: Any = None)`
     - **Function**
        Initializes the `Plan` object. If `plan_info` is provided, constructs the corresponding `PlanGraph`.
     - **Parameter**
       - `plan_info`: Structured information (type may vary) describing tasks and dependencies.
  2. `create_graph_from_plan_info(plan_info: Any) -> None`
     - **Function**
        Parses `plan_info` and creates nodes and edges for the `PlanGraph`.
     - **Parameter**
       - `plan_info`: Object describing tasks and their dependency structure.
  3. `prepare_tools() -> None`
     - **Function**
        Preloads and allocates all `Tool` instances required by the execution plan. If a tool is by default in CPU mode, tries to move it to an available GPU.
  4. `clean_tools() -> None`
     - **Function**
        Frees or unloads the used tools, moves them back to CPU, and clears caches.
  5. `_execute_on_graph(input_data: Any, cost_aware: bool) -> None`
     - **Function**
        Executes each node in topological or dependency order, storing the result in each node.
     - **Parameters**
       - `input_data`: Initial input data for the starting node.
       - `cost_aware`: Whether to track resource and cost statistics.
  6. `collect_results() -> Dict[TaskName, Any]`
     - **Function**
        Collects outputs from all end-point nodes in the plan and returns them as a dictionary.
  7. `calculate_price_and_save() -> float`
     - **Function**
        Calculates the accumulated cost of the entire plan based on each node’s cost information and predefined pricing configurations, saving the value to `self.price`.
  8. `calculate_exec_time_and_save() -> float`
     - **Function**
        Uses a BFS-like method to compute the total execution time of the plan’s critical path and saves it to `self.exec_time`.
  9. `execute(input_data: Any, cost_aware: bool = True) -> Any`
     - **Function**
        Calls a series of methods (`prepare_tools()`, `_execute_on_graph()`, `collect_results()`, `calculate_exec_time_and_save()`, `calculate_price_and_save()`, etc.) to run the entire plan and return the output.
     - **Parameters**
       - `input_data`: Input data for the plan’s starting node.
       - `cost_aware`: Whether to track and compute execution costs.
  10. `cleanup(clean_tools: bool = True) -> None`
      - **Function**
         Cleans up the `Plan` object and frees resources, resetting its graph, tools, pricing, and execution time.
      - **Parameter**
        - `clean_tools`: Whether to call `clean_tools()` first.

#### 2. `src/plan_graph.py`

#### Main Class

#### `PlanGraph`

- **Purpose**  
  Maintains a directed graph structure, including the methods for managing nodes and edges such as adding, removing, etc.

- **Primary Attributes**  
  - `name_to_id`: A dictionary mapping task names (`TaskName`) to node IDs.  
  - `nodes`: A dictionary with `node_id` as the key and the corresponding `PlanNode` instance as the value.  
  - `edges`: A dictionary with `edge_id` as the key and the corresponding `PlanEdge` instance as the value.

- **Key Methods**  
  1. `__init__()`  
     - **Function**  
       Initializes the graph structure, automatically creating a default start node (`DEFAULT_START_TASK_NAME`).
  2. `start_node` (property)  
     - **Function**  
       Returns the default start node (usually with `node_id=0`).
  3. `add_node(...) -> PlanNode`  
     - **Function**  
       Adds a new `PlanNode` to the graph and registers the mapping from `task_name` to `node_id`.  
     - **Key Parameters**  
       - `task_name: TaskName`  
       - `model_name: Optional[ModelName]`  
       - `is_start_point: bool`  
       - `is_end_point: bool`  
       - ...
  4. `get_or_add_node(task_name: TaskName) -> PlanNode`  
     - **Function**  
       Retrieves an existing node by task name; if it doesn’t exist, creates a new one.
  5. `add_edge(source: PlanNode, target: PlanNode) -> PlanEdge`  
     - **Function**  
       Creates a directed edge from `source` to `target` in the graph, and updates both nodes accordingly.
  6. `remove_node(node_id: NodeID) -> None`  
     - **Function**  
       Removes the node with the specified `node_id` and all associated edges.
  7. `remove_edge(edge_id: EdgeID) -> None`  
     - **Function**  
       Removes the edge with the specified `edge_id` and updates the source and target nodes accordingly.

#### `PlanNode`

- **Purpose**  
  Represents a node in the `PlanGraph`, containing task name, model name, data, execution info, etc.

- **Primary Attributes**  
  - `node_id`: Unique node ID.  
  - `task_name`: Name of the task for this node.  
  - `model_name`: Optional name of the model.  
  - `is_start_point`: Whether this node is a start node.  
  - `is_end_point`: Whether this node is an end node.  
  - `value`: The stored value/result of the node’s execution.  
  - `costs`: Execution cost information for this node (see `CostInfo`).  
  - `price`: The price of executing the node.  
  - `critical_exec_time`: The accumulated runtime on the critical path.  
  - `in_edges`, `out_edges`: Weak-reference dictionaries for incoming and outgoing edges, respectively.

- **Key Methods**  
  1. `__init__(...)`  
     - **Function**  
       Initializes the node with various attributes and sets up empty `in_edges` and `out_edges`.  
  2. `get_value() -> Any`  
     - **Function**  
       Returns the current stored result for the node.  
  3. `set_value(value: Any) -> None`  
     - **Function**  
       Stores a result in the node and marks the node as completed.  
  4. `calculate_price_and_save() -> float`  
     - **Function**  
       Calculates the node’s price based on cost data and config (`Mcfg`), saves it to `self.price`, and returns the value.

#### `PlanEdge`

- **Purpose**  
  A directed edge in the graph, storing weak references to the source and target nodes, as well as a reference to the parent graph.

- **Primary Attributes**  
  - `edge_id`: Unique edge ID.  
  - `source`: Weak reference to the source node.  
  - `target`: Weak reference to the target node.

- **Key Methods**  
  - `__init__(edge_id: EdgeID, source: PlanNode, target: PlanNode)`  
    - **Function**  
      Initializes the edge with the given `edge_id` and source/target references.

------

#### 3. `src/tool/tool.py`

#### Main Class

#### `Tool`

- **Purpose**  
  Wraps a single model or executable process, optionally monitoring resource usage (CPU/GPU usage, execution time, etc.).

- **Primary Attributes**  
  - `config: ModelConfig`: Configuration info for the model.  
  - `model: Optional[torch.nn.Module]`: A PyTorch model instance.  
  - `process: Optional[Callable[..., Any]]`: A callable for inference or other custom operations.  
  - `options: Dict[str, Any]`: Extra initialization options.  
  - `_device: str`: The current device (e.g., `cpu` or `cuda:0`) on which the model resides.

- **Key Methods**  
  1. `__init__(config, model, process=None, device='cpu', **kwargs)`  
     - **Function**  
       Records the model and its configuration, moves it to the specified device, and sets it to eval mode.  
     - **Parameters**  
       - `config: ModelConfig`  
       - `model: torch.nn.Module`  
       - `process: Optional[Callable[..., Any]]`  
       - `device: str`  
       - `kwargs: Any` (additional options)
  2. `device` (property) & setter  
     - **Function**  
       Gets or sets the model’s current device; the setter moves the model to that device.
  3. `to(device: str) -> None`  
     - **Function**  
       Manually moves the model to the specified device.
  4. `execute(*args: Any, cost_aware: bool, **kwargs: Any) -> Any`  
     - **Function**  
       Runs the `process` function. If `cost_aware` is true, gathers CPU/GPU memory usage and execution time metrics, then returns both the result and metric info.  
     - **Parameters**  
       - `cost_aware: bool`  
       - Other `*args` and `**kwargs` are passed to the `process`.

------

#### 4. `src/tool/tool_manager.py`

#### Main Class

#### `ToolManager`

- **Purpose**  
  Manages various tools (`Tool`) for different task types (`TaskName`), responsible for loading, listing, and retrieving tool instances for specified tasks/models.

- **Primary Attributes**  
  - `tool_cls_groups: Dict[TaskName, Type[GroupedTools]]`: Maps task names to their corresponding grouped tool classes.  
  - `tool_groups: Dict[TaskName, GroupedTools]`: Stores instantiated grouped tools at runtime.

- **Key Methods**  
  1. `__init__()`  
     - **Function**  
       Initializes an empty `tool_groups` dictionary.
  2. `load_model(task_name: TaskName, model_name: ModelName) -> None`  
     - **Function**  
       Loads the tool group for the specified task/model, handling internal initialization or caching of the model.
  3. `load_models(task_name: TaskName = 'all_tasks', model_name: ModelName = 'all_models') -> None`  
     - **Function**  
       Depending on the parameters, loads all models for all tasks, or only specific tasks/models.
  4. `list_models() -> Dict[TaskName, List[ModelName]]`  
     - **Function**  
       Compiles a list of all loaded models grouped by task, returning a dictionary keyed by task name.
  5. `get_model(task_name: TaskName, model_name: ModelName) -> Tool`  
     - **Function**  
       Retrieves a loaded `Tool` instance for the specified task/model. If `model_name` is not provided, uses the default (the first model in `MODEL_REGISTRY`).

------

#### 5. `src/dataloader.py`

#### Main Class

#### `TaskDataset`

- **Purpose**  
  A PyTorch-style `Dataset` for loading both image and text task data, with samples indexed by `sample_id`.

- **Primary Attributes**  
  - `input_data`: A dictionary keyed by `sample_id`, with each value being the corresponding image/tensor or text content.  
  - `output_data`: A dictionary of outputs keyed by the same `sample_id`.  
  - `sample_ids`: A list of sample IDs for indexing.

- **Key Methods**  
  1. `__init__(data_path: str, *, task_id: int)`  
     - **Function**  
       Initializes and loads data for the specified task using `data_path` and `task_id`.  
     - **Parameters**  
       - `data_path: str`  
       - `task_id: int`
  2. `_load_images(image_dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor]]`  
     - **Function**  
       Loads image files from the specified path, converts them to tensors, and stores them by `sample_id`.
  3. `_load_text(file_path: str) -> Dict[SampleID, Dict[str, TextContent]]`  
     - **Function**  
       Loads each line from a text file, mapping line numbers to `sample_id`.
  4. `_load_files(dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor | TextContent]]`  
     - **Function**  
       Identifies `images` folders or `.txt` files in the specified directory, calls the respective loading methods, and merges the data by sample ID.
  5. `_load_data() -> None`  
     - **Function**  
       Reads data from the input/output paths and updates `input_data` and `output_data`.
  6. `__getitem__(index: int) -> Dict[str, int | Dict[str, torch.Tensor | TextContent]]`  
     - **Function**  
       Returns the complete data for a single sample (including sample ID, input, and output) by index.
  7. `__len__() -> int`  
     - **Function**  
       Returns the number of samples in the dataset.

### Part 2: CATP-LLM

The main codes of CATP-LLM are listed below:

```
./src
├── catpllm
│   ├── data 
│   |   ├── config_data
│   |   ├── training_data
│   |   ├── plan_dataset.py
│   |   ├── plan_pool.py
│   ├── model
│   |   ├── llm
│   |   ├── __init__.py
│   |   ├── offline_rl.py
│   |   ├── prompt.py
│   |   ├── token_encoder.py
│   |   ├── tokens.py
│   ├── pipeline
│   |   ├── inference.py
│   |   ├── test.py
│   |   ├── train.py
│   ├── utils
│   |   ├── cost_utils.py
│   |   ├── llm_utils.py
│   |   ├── utils.py
├── run_catpllm.py
```
Some key code directory or files are explained as follows.
#### 1. `data`
This directory stores the CATP-LLM related data or data processing codes.
- `config_data`: Store the configuration files of running CATP-LLM.
- `training_data`: Store the data files of training CATP-LLM on sequential or non-sequential tasks.
- `plan_pool.py`: Implement a class `PlanPool` to store the plans for training CATP-LLM.
- `plan_dataset.py`: Implement a class `PlanDataset` to wrap the `PlanPool` for training CATP-LLM.

#### 2. `model`
This directory stores the key modules of CATP-LLM.
- `llm`: Store the codes of LLMs. Most of them are copied from Huggingface Transformer.
- `tokens.py`: Implement the tool and dependency tokens of CATP-LLM.
- `token_encoder.py`: Implement the token encoder  of CATP-LLM to encode tool and dependency tokens. 
- `prompt.py`: Store the prompts of CATP-LLM.
- `offline_rl.py`: Implement the offline RL policy model of CATP-LLM.

#### 3. `pipeline`
This directory stores the key pipelines of CATP-LLM.
- `train.py`: Implement the training pipeline of CATP-LLM.
- `inference.py`: Implement the inference pipeline of CATP-LLM to generate a tool plan.
- `test.py`: Implement the testing pipeline of CATP-LLM, which calls the `inference.py` to generate tool plans and evaluate them.

## How to add new baselines to this repo?
Basically, there are simply three steps to add a new baseline to this repo:
1. Create a new directory `src/your_baseline`, and place all the codes related to this baseline in this directory, just like `src/captllm`.
2. Write a function to transform the tool plan format of your baseline into the OpenCATP format. 

   OpenCATP describes a tool plan in the form of `[Tool A, [Dependency of Tool A], Tool B, [Dependency of Tool B], ...]`. To be compatible with the codes of OpenCATP, you need to make sure the tool plans generated by your baseline follow this format. Please refer to `src/catpllm/utils/utils.token_plan_to_opencatp_plan` for how to write this transformation function.

3. Write a new entry file `run_your_baseline.py` for running your baseline, just like `run_catpllm.py`.

## Citation
If you find this repository useful, please cite our paper:

> @inproceedings{wu2025catp,
>       author = {Wu, Duo and Wang, Jinghe and Meng, Yuan and Zhang, Yanning and Sun, Le and Wang, Zhi},
>       title = {CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning},
>       year = {2025},
>       booktitle = {ICCV}
> }

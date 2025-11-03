import json
import sys
import os
import argparse

# from torch.utils.data import DataLoader
from torch.utils.data import default_collate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TOOL_DEVICE_LIST, EVALUATOR_DEVICE_LIST
from src.plan.plan import Plan
from src.data_loader import TaskDataset
from src.metrics.evaluator import calculate_qop, calculate_task_score

from src.tools.tool_manager import tool_manager
# import pickle

tool_manager.load_models()

with open("./seq_plan_pool_with_tools.json", "r") as f:
    seq_plan_pool_with_tools = json.load(f)

with open("./nonseq_plan_pool_with_tools.json", "r") as f:
    nonseq_plan_pool_with_tools = json.load(f)

TOOL_DEVICE_LIST = ['cuda:4']
EVALUATOR_DEVICE_LIST = ['cuda:5']
sid = 0
nonseq_plans = {}
for task_id, task_plans in nonseq_plan_pool_with_tools["plans"].items():
    print(f"now execute {task_id}.")
    task_id = int(task_id)
    # valid_task_id_range = task_id_range[process_id]
    # if not (valid_task_id_range[0] <= task_id <= valid_task_id_range[1]):
    #     continue
    dataset = TaskDataset("../dataset", task_id=int(task_id))

    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # sample = next(iter(data_loader))
    nonseq_plans[task_id] = {}
    for sample_id, plans in task_plans.items():

        nonseq_plans[task_id][sample_id] = []
        data = dataset[int(sample_id)]
        data = default_collate([data])

        input_data = data["input"]
        output_data = data["output"]

        for plan_info in plans:
            plan = Plan(plan_info)
            result = plan.execute(input_data)

            sid += 1
            print(sid)

            exec_time_list = []
            short_term_cpu_memory_list = []
            short_term_gpu_memory_list = []
            for i in range(0, len(plan_info), 2):
                tool_name = plan_info[i]
                model_name, _ = tool_name.rsplit("_", 1)
                tool_id = plan.graph.name_to_id[model_name]
                tool = plan.graph.nodes[tool_id]
                costs = tool.costs
                exec_time, short_term_cpu_memory, short_term_gpu_memory = (
                    costs["exec_time"],
                    costs["short_term_cpu_memory"],
                    costs["short_term_gpu_memory"],
                )
                exec_time_list.append(exec_time)
                short_term_cpu_memory_list.append(short_term_cpu_memory)
                short_term_gpu_memory_list.append(short_term_gpu_memory)

            avg_score = calculate_task_score(result, output_data, sequential=False)
            qop = calculate_qop(avg_score, plan.price)

            total_info = {
                "exec_time": plan.exec_time,
                "price": plan.price,
                "avg_score": avg_score,
                "qop": qop,
            }

            nonseq_plans[task_id][sample_id].append(
                {
                    "task_id": task_id,
                    "sample_id": sample_id,
                    "plan_info": plan_info,
                    "exec_time_list": exec_time_list,
                    "short_term_cpu_memory_list": short_term_cpu_memory_list,
                    "short_term_gpu_memory_list": short_term_gpu_memory_list,
                    "total_info": total_info,
                }
            )
            
            if sid % 500 == 0:
                with open(f"nonseq_result_{sid}.json", "w") as f:
                    json.dump(nonseq_plans, f, indent=4)
            # print(result)

print("nonseq_plans executed")

with open("./nonseq_plans.json", "w") as f:
    json.dump(nonseq_plans, f, indent=4)

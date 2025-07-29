import json

from torch.utils.data import DataLoader

from src.config import GlobalPathConfig
from src.plan.plan import Plan
from src.data_loader import TaskDataset
from src.metrics.evaluator import calculate_qop, calculate_task_score
from src.tools.tool_manager import tool_manager
import pickle

# with open("./test_samples.json", "r") as f:
#     data = json.load(f)


# for task_id in data.keys():
#     plan = data[task_id]["plan"]
#     plan = Plan(plan)
#     task_id = int(task_id)
#     data_set = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
#     data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
#     for batch in data_loader:
#         sample_id = batch["sample_id"]
#         input_data = batch["input"]
#         output_data = batch["output"]
#         result = plan.execute(input_data)
#         if result is None:
#             pass
#         else:
#             task_score = calculate_task_score(result, output_data, sequential=task_id < 200)
#             cost_price = plan.price
#             exec_time = plan.exec_time
#             qop = calculate_qop(task_score, cost_price)
#             print(task_id, task_score, cost_price, exec_time, qop)
#         break
# print("done")
# import jsonpickle

# with open('./src/catpllm/data/training_data/seq_plan_pool.pkl', 'rb') as f:
#     seq_plan_pool = pickle.load(f)
#     seq_plan_pool_json = jsonpickle.encode(seq_plan_pool, unpicklable=False)
#     with open('./seq_plan_pool.json', 'w') as json_file:
#         json_file.write(seq_plan_pool_json)

# with open('./src/catpllm/data/training_data/nonseq_plan_pool.pkl', 'rb') as f:
#     nonseq_plan_pool = pickle.load(f)
#     nonseq_plan_pool_json = jsonpickle.encode(nonseq_plan_pool, unpicklable=False)
#     with open('./nonseq_plan_pool.json', 'w') as json_file:
#         json_file.write(nonseq_plan_pool_json)

# print('done')

# tool_manager.load_models()


with open("./seq_plan_pool_with_tools.json", "r") as f:
    seq_plan_pool_with_tools = json.load(f)

with open("./nonseq_plan_pool_with_tools.json", "r") as f:
    nonseq_plan_pool_with_tools = json.load(f)

for task_id, task_plans in seq_plan_pool_with_tools['plans'].items():
    dataset = TaskDataset('../dataset', task_id=task_id)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    sample = next(iter(data_loader))

    for sample_id, plans in task_plans.items():
        data = dataset[sample_id]
        for plan_info in plans:
            plan = Plan(plan_info)
            plan.execute(data)
            print('test')

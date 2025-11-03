from copy import deepcopy
import json


def merge_two_plans_dict(d1: dict, d2: dict) -> dict:
    result = deepcopy(d2)

    for task_id, samples in d1.items():
        if task_id not in result:
            result[task_id] = deepcopy(samples)
            continue

        for sample_id, value in samples.items():
            if sample_id not in result[task_id]:
                result[task_id][sample_id] = deepcopy(value)
                continue

            target_val = result[task_id][sample_id]

            assert isinstance(target_val, list) and isinstance(value, list)
            all_value = target_val + value
            result[task_id][sample_id] = all_value

    return result


with open("./results/seq_result_1000.json", "r") as f:
    data_1 = json.load(f)

with open("./results/seq_plans.json", "r") as f:
    data_2 = json.load(f)


seq_data = merge_two_plans_dict(data_1, data_2)

with open("./results/seq_plan_pool.json", "w") as f:
    json.dump(seq_data, f)



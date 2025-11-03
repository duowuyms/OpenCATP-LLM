import json
import random
from copy import deepcopy

random.seed(42)

def rename_tasks_in_one_plan(plan):
    task_old2new = {}
    renamed_plan = []

    for i in range(0, len(plan), 2):
        task = plan[i]
        if task not in task_old2new: # todo Fix this?
            suffix = random.choice([1, 2, 3])
            task_old2new[task] = f"{task}_{suffix}"
            if task == 'image_super_resolution':
                task_old2new[task] = 'image_super_resolution_1'

    for i in range(0, len(plan), 2):
        old_task = plan[i]
        deps = plan[i + 1]

        assert len(deps) == 1

        new_task = task_old2new[old_task]
        new_deps = [task_old2new.get(d, d) for d in deps]  # input_of_query don't change

        renamed_plan.extend([new_task, new_deps])

    return renamed_plan


def expand_plans(task_plans):
    new_task_plans = {}
    for sample_id, plans in task_plans.items():
        new_plans = []
        for plan in plans:
            for _ in range(3):
                new_plans.append(rename_tasks_in_one_plan(plan))
        new_task_plans[sample_id] = new_plans
    return new_task_plans


if __name__ == "__main__":
    with open("./seq_plan_pool.json") as f:
        origin_seq_plans = json.load(f)["plans"]

    with open("./nonseq_plan_pool.json") as f:
        origin_nonseq_plans = json.load(f)["plans"]

    new_seq_plans = {}
    for task_id, task_plans in origin_seq_plans.items():
        if task_id == 'default_factory':
            continue
        new_seq_plans[task_id] = expand_plans(task_plans)

    new_nonseq_plans = {}
    for task_id, task_plans in origin_nonseq_plans.items():
        if task_id == 'default_factory':
            continue
        new_nonseq_plans[task_id] = expand_plans(task_plans)

    with open("./seq_plan_pool_with_tools.json", "w") as f:
        json.dump({"plans": new_seq_plans}, f, indent=2, ensure_ascii=False)

    with open("./nonseq_plan_pool_with_tools.json", "w") as f:
        json.dump({"plans": new_nonseq_plans}, f, indent=2, ensure_ascii=False)

    print("Done!")

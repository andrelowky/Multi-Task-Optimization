import torch
import numpy as np
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def calc_hv(y, task, hv, problems, negate=True, hv_scale=10000):
    volume_list = []
    for i, problem in enumerate(problems):
        specific_y = y[(task == i).all(dim=1)]
            
        pareto_mask = is_non_dominated(specific_y)
        pareto_y = specific_y[pareto_mask]
        volume_list.append(hv.compute(pareto_y)/problems[0].hv_scaling)

    return np.array(volume_list)

def update_values(old_tensors, new_tensors, problems):
    old_x, old_task, old_y = old_tensors
    new_x, new_task = new_tensors

    for i, problem in enumerate(problems):
        added_x = new_x[(new_task==i).all(dim=1)]
        added_task = new_task[(new_task==i).all(dim=1)]
        added_y = problem.evaluate(added_x)
    
        old_x = torch.cat([old_x, added_x])
        old_task = torch.cat([old_task, added_task])
        old_y = torch.cat([old_y, added_y])

    return old_x, old_task, old_y


def calc_losses(model, mll):
    with torch.no_grad():
        outputs = model(*model.train_inputs)
        loss = -mll(outputs, model.train_targets).item()

    return abs(loss)
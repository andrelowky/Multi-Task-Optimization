import torch
import math
import numpy as np

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list, optimize_acqf_mixed
from botorch.utils.multi_objective.pareto import is_non_dominated

import pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination

raw_samples = 256

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def optimize_list(acq_func_list, bounds):
	# for qnparego and qucb
	
	candidates, _ = optimize_acqf_list(
		acq_function_list=acq_func_list,
		bounds=bounds,
		num_restarts=2,
		raw_samples=raw_samples,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_fixed_list(acq_func_list, bounds, n_task):
	# for qnparego and qucb
	
	candidates, _ = optimize_acqf_list(
		acq_function_list=acq_func_list,
		fixed_features_list=[{bounds.shape[1]-1: task_idx} for task_idx in range(n_task)],
		bounds=bounds,
		num_restarts=2,
		raw_samples=raw_samples,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_mt_mixed(acq_func, acq_bounds, batch_size, n_task):
	# for mt qnehvi
	
	candidates, _ = optimize_acqf_mixed(
		acq_function=acq_func,
		bounds=acq_bounds,
		q=batch_size,
		fixed_features_list=[{bounds.shape[1]-1: task_idx} for task_idx in range(n_task)],
		num_restarts=2,
		raw_samples=raw_samples,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates


def optimize_st_acqf(acq_func, batch_size, std_bounds):
	# for st qnehvi and qucb
	
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=std_bounds,
        q=batch_size,
        num_restarts=2,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    return candidates


def optimize_st_egbo(acq_func, x, y, batch_size):
	# for st qnehvi

	# we pick out the best points so far to form parents
    pareto_mask = is_non_dominated(y)
    pareto_y = -y[pareto_mask]
    pareto_x = x[pareto_mask]
	
    nsga = UNSGA3(
		pop_size=raw_samples,
		ref_dirs=get_reference_directions("energy", y.shape[1], batch_size),
	)
	
    pymooproblem = PymooProblem(
		n_var=x.shape[1], n_obj=y.shape[1], 
		xl=np.zeros(x.shape[1]), xu=np.ones(x.shape[1])
	)

    nsga.setup(pymooproblem, termination=NoTermination())

	# set the 1st population to the current evaluated population
    pop = Population.new("X", pareto_x.cpu().numpy()).set("F", pareto_y.cpu().numpy())
    nsga.tell(infills=pop)

	# propose children based on tournament selection -> crossover/mutation
    newpop = nsga.ask()
    nsga3_x = torch.tensor(newpop.get("X"), **tkwargs)
	
	##########
	
    acq_value_list = []

    for i in range(0, nsga3_x.shape[0]):
        with torch.no_grad():
            acq_value = acq_func(nsga3_x[i].unsqueeze(dim=0))
            acq_value_list.append(acq_value.item())

    sorted_x = nsga3_x.cpu().numpy()[np.argsort(acq_value_list)]

	##########
	
    return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples

def optimize_mt_egbo(acq_func, x, task, y, batch_size, n_obj):
	# for mt qnehvi
	n_task = int(task.max())
	
	# we pick out the best points so far to form parents, from each task
	pareto_mask = is_non_dominated(y[(task==0).all(dim=1)])
	pareto_y = -y[(task==0).all(dim=1)][pareto_mask].cpu().numpy() # i flip back to maximization
	pareto_x = x[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	pareto_task = task[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	
	for i in range(1, n_task):
		pareto_mask = is_non_dominated(-y[(task==i).all(dim=1)])
		pareto_y = np.vstack([pareto_y, -y[(task==i).all(dim=1)][pareto_mask].cpu().numpy()]) # same here
		pareto_x = np.vstack([pareto_x, x[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
		pareto_task = np.vstack([pareto_task, task[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
	
	nsga = UNSGA3(
		pop_size=raw_samples,
		ref_dirs=get_reference_directions("energy", n_obj, batch_size),
	)
	
	xl = np.zeros(x.shape[1]+1)
	xu = np.ones(x.shape[1]+1)
	xu[-1] = int(task.max())
	
	pymooproblem = PymooProblem(n_var=x.shape[1], n_obj=len(ref_pt), xl=xl, xu=xu)
	nsga.setup(pymooproblem, termination=NoTermination())
	
	# set the 1st population to the current evaluated population
	pop = Population.new("X", np.hstack([pareto_x, pareto_task])).set("F", pareto_y)
	nsga.tell(infills=pop)
	
	# propose children based on tournament selection -> crossover/mutation
	newpop = nsga.ask()
	candidates = torch.tensor(newpop.get("X"), **tkwargs)
	candidates[:, -1] =  torch.floor(candidates[:, -1]) # round down last column which is task index
	
	##########
	
	acq_value_list = []
	
	for i in range(0, candidates.shape[0]):
		with torch.no_grad():
			acq_value = acq_func(candidates[i].unsqueeze(dim=0))
			acq_value_list.append(acq_value.item())
	
	sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]
	
	##########
	
	return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples
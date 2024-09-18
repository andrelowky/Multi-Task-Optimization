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

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def optimize_mt_list(acq_func_list, acq_bounds):
	# for qnparego
	
	candidates, _ = optimize_acqf_list(
		acq_function_list=acq_func_list,
		bounds=acq_bounds,
		fixed_features_list=[{acq_bounds.shape[1]-1: task_idx} 
							 for task_idx in range(int(acq_bounds[1, -1])+1)],
		num_restarts=2,
		raw_samples=128,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_mt_mixed(acq_func, acq_bounds, batch_size):
	# for mt qnehvi and qucb
	
	candidates, _ = optimize_acqf_mixed(
		acq_function=acq_func,
		bounds=acq_bounds,
		q=batch_size,
		fixed_features_list=[{acq_bounds.shape[1]-1: task_idx} 
							 for task_idx in range(int(acq_bounds[1, -1])+1)],
		num_restarts=2,
		raw_samples=128,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_st_list(acq_func_list, std_bounds):
	# for st qnparego
	
	candidates, _ = optimize_acqf_list(
		acq_function_list=acq_func_list,
		bounds=std_bounds,
		num_restarts=2,
		raw_samples=128,
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
        raw_samples=128,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    return candidates


def optimize_st_egbo(acq_func, ref_pt, x, y, batch_size, n_sampling=256):
	# for st qnehvi

	# we pick out the best points so far to form parents
    pareto_mask = is_non_dominated(y)
    pareto_y = -y[pareto_mask]
    pareto_x = x[pareto_mask]
	
    nsga = UNSGA3(
		pop_size=n_sampling,
		ref_dirs=get_reference_directions("energy", len(ref_pt), batch_size),
		sampling=pareto_x.cpu().numpy()
	)
	
    pymooproblem = PymooProblem(
		n_var=x.shape[1], n_obj=len(ref_pt), 
		xl=np.zeros(x.shape[1]), xu=np.ones(x.shape[1])
	)

    nsga.setup(pymooproblem, termination=NoTermination())

    print(pareto_y.shape)
	# set the 1st population to the current evaluated population
	pop = Population.new("X", pareto_x)
	pop.set("F", pareto_y.cpu().numpy())
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

def optimize_mt_egbo(acq_func, ref_pt, x, task, y, batch_size, n_sampling=256):
	# for mt qnehvi

	# we pick out the best points so far to form parents
    n_task = int(task.max())+1
    
    pareto_mask = is_non_dominated(y[(task==0).all(dim=1)])
    pareto_y = -y[(task==0).all(dim=1)][pareto_mask].cpu().numpy() # i flip back to maximization
    pareto_x = x[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
    
    for i in range(1, n_task+1):
        pareto_mask = is_non_dominated(-y[(task==i).all(dim=1)])
        pareto_y = np.vstack((pareto_y, -y[(task==i).all(dim=1)][pareto_mask].cpu().numpy())) # same here
        pareto_x = np.vstack((pareto_x, x[(task==i).all(dim=1)][pareto_mask].cpu().numpy()))

    nsga = UNSGA3(
		pop_size=n_sampling,
		ref_dirs=get_reference_directions("energy", len(ref_pt), batch_size),
		sampling=pareto_x,
	)

    pymooproblem = PymooProblem(n_var=x.shape[1], n_obj=len(ref_pt), 
								xl=np.zeros(x.shape[1]),
								xu=np.ones(x.shape[1]))

    nsga.setup(pymooproblem, termination=NoTermination())

	# set the 1st population to the current evaluated population
    pop = Population.new("X", pareto_x)
    pop.set("F", pareto_y)
    nsga.tell(infills=pop)

	# propose children based on tournament selection -> crossover/mutation
    newpop = nsga.ask()
    nsga3_x = torch.tensor(newpop.get("X"), **tkwargs)

	# fill them back up with task index
    candidates = nsga3_x.tile(n_task, 1)
    candidate_task = torch.tensor(0, **tkwargs).repeat(n_sampling).reshape(-1,1)
    for i in range(1, n_task):
	    candidate_task = torch.vstack([candidate_task,
	                                   torch.tensor(i, **tkwargs).repeat(n_sampling).reshape(-1,1)])
	
    candidates = torch.hstack([candidates, candidate_task])
	
	##########
	
    acq_value_list = []

    for i in range(0, candidates.shape[0]):
        with torch.no_grad():
            acq_value = acq_func(candidates[i].unsqueeze(dim=0))
            acq_value_list.append(acq_value.item())

    sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]

	##########
	
    return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples
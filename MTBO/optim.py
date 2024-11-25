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

from pymoo.core.individual import Individual
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.util.ref_dirs import get_reference_directions
from sklearn.preprocessing import MinMaxScaler

from botorch.utils.sampling import draw_sobol_samples

raw_samples = 512

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def optimize_st_list(acq_func_list, bounds):
	# for qnparego and qucb
	
	candidates, _ = optimize_acqf_list(
		acq_function_list=acq_func_list,
		bounds=bounds,
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
	
	pareto_mask = is_non_dominated(y)
	pareto_x = x[pareto_mask].cpu().numpy()
	
	pymooproblem = PymooProblem(n_var=x.shape[1], xl=np.zeros(x.shape[1]), xu=np.ones(x.shape[1]))
	
	parents = [
		[Individual(X=pareto_x[np.random.choice(pareto_x.shape[0])]),
		 Individual(X=pareto_x[np.random.choice(pareto_x.shape[0])])] 
	for _ in range(int(raw_samples/2))]
	
	off = SBX().do(pymooproblem, parents)
	candidates = torch.tensor(off.get("X"), **tkwargs)
		
	acq_value_list = [acq_func(candidates[i].unsqueeze(dim=0)).detach().item()
					  for i in range(candidates.shape[0])]
	sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]
	
	return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples

def optimize_mt_mixed_flexi(acq_func, acq_bounds, batch_size, n_task):
	# for mt qnehvi
	
	candidates, _ = optimize_acqf_mixed(
		acq_function=acq_func,
		bounds=acq_bounds,
		q=batch_size,
		fixed_features_list=[{acq_bounds.shape[1]-1: task_idx} for task_idx in range(n_task)],
		num_restarts=2,
		raw_samples=raw_samples,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_mt_mixed_fixed(acq_func, acq_bounds, batch_size, task_idx):
	# for mt qnehvi
	
	candidates, _ = optimize_acqf_mixed(
		acq_function=acq_func,
		bounds=acq_bounds,
		q=batch_size,
		fixed_features_list=[{acq_bounds.shape[1]-1: task_idx}],
		num_restarts=2,
		raw_samples=raw_samples,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_mt_list_flexi(acq_func_list, bounds, n_task):
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

def optimize_mt_list_fixed(acq_func_list, bounds, task_idx):
	# for qnparego and qucb
	
	candidates, _ = optimize_acqf_list(
		acq_function_list=acq_func_list,
		fixed_features_list=[{bounds.shape[1]-1: task_idx}],
		bounds=bounds,
		num_restarts=2,
		raw_samples=raw_samples,
		options={"batch_limit": 5, "maxiter": 200},
		)

	return candidates

def optimize_mt_egbo_flexi(acq_func, x, task, y, batch_size):
	n_tasks = int(task.max()+1)
	
	pareto_mask = is_non_dominated(y[(task==0).all(dim=1)])
	pareto_x = x[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	pareto_task = task[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	
	for i in range(1, int(task.max())+1):
		pareto_mask = is_non_dominated(y[(task==i).all(dim=1)])
		pareto_x = np.vstack([pareto_x, x[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
		pareto_task = np.vstack([pareto_task, task[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
	
	pareto_x_task = np.hstack([pareto_x, pareto_task])
	
	xl=np.zeros(x.shape[1]+1)
	xu=np.ones(x.shape[1]+1)
	xu[-1] = int(task.max())
	pymooproblem = PymooProblem(n_var=x.shape[1]+1, xl=xl, xu=xu)
	
	parents = [
		[Individual(X=pareto_x_task[np.random.choice(pareto_x_task.shape[0])]),
		 Individual(X=pareto_x_task[np.random.choice(pareto_x_task.shape[0])])] 
	for _ in range(int(raw_samples/2))]
	
	off = SBX().do(pymooproblem, parents)
	candidates = torch.tensor(off.get("X"), **tkwargs)
	candidates[:,-1] = torch.floor(candidates[:,-1]) # bring back to integer
	
	acq_value_list = [acq_func(candidates[i].unsqueeze(dim=0)).detach().item()
					  for i in range(candidates.shape[0])]
	
	return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples

def optimize_mt_egbo_fixed(acq_func, x, task, y, batch_size):
	n_tasks = int(task.max()+1)
	
	pareto_mask = is_non_dominated(y[(task==0).all(dim=1)])
	pareto_x = x[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	
	for i in range(1, int(task.max())+1):
		pareto_mask = is_non_dominated(y[(task==i).all(dim=1)])
		pareto_x = np.vstack([pareto_x, x[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
	
	pymooproblem = PymooProblem(n_var=x.shape[1], xl=np.zeros(x.shape[1]), xu=np.ones(x.shape[1]))
	
	parents = [
		[Individual(X=pareto_x[np.random.choice(pareto_x.shape[0])]),
		 Individual(X=pareto_x[np.random.choice(pareto_x.shape[0])])] 
	for _ in range(int(raw_samples/2))]
	
	off = SBX().do(pymooproblem, parents)
	candidates_x = torch.tensor(off.get("X"), **tkwargs)
	
	all_candidates = []
	for task_idx in range(n_tasks):
		candidates_i = torch.hstack([candidates_x, 
									 torch.tensor(task_idx).repeat(candidates_x.shape[0], 1)
												  ])
		acq_value_list = [acq_func(candidates_i[i].unsqueeze(dim=0)).detach().item()
						  for i in range(candidates_i.shape[0])]
		all_candidates.append(candidates_i.cpu().numpy()[np.argsort(acq_value_list)][-batch_size:])
	
	return torch.tensor(all_candidates, **tkwargs)

def optimize_mt_sobol(acq_func, batch_size, acq_bounds):

    candidates = draw_sobol_samples(bounds=acq_bounds, n=raw_samples, q=1).squeeze(1)
    candidates[:,-1] = torch.round(candidates[:,-1])

    acq_value_list = [acq_func(candidates[i].unsqueeze(dim=0)).detach().item()
                      for i in range(candidates.shape[0])]
    sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]

    return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples
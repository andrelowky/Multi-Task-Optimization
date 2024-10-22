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

raw_samples = 256

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

def optimize_mt_egbo(acq_func, x, task, y, batch_size):
	n_tasks = int(task.max()+1)
	
	pareto_mask = is_non_dominated(y[(task==0).all(dim=1)])
	pareto_x = x[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	pareto_task = task[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	pareto_y = y[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	
	for i in range(1, int(task.max())+1):
		pareto_mask = is_non_dominated(y[(task==i).all(dim=1)])
		pareto_x = np.vstack([pareto_x, x[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
		pareto_task = np.vstack([pareto_task, task[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
		pareto_y = np.vstack([pareto_y, y[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
	
	
	ref_dirs = get_reference_directions("energy", y.shape[1]+1, n_tasks)
	for i in range(y.shape[1]):
		if y[0,i] < 0: # ensure correct direction if we have minimization objectives
			ref_dirs[:,i] = -ref_dirs[:,i]
	
	# we select for pareto solutions which are spread in both y and task space
	scaler = MinMaxScaler()
	pareto_y_task = scaler.fit_transform(np.hstack([pareto_y, pareto_task]))
	pareto_x_task = np.hstack([pareto_x, pareto_task])
	
	def assign_ref(ref_vectors, points):
		distances = np.linalg.norm(points[:, np.newaxis] - ref_vectors, axis=2)
		nearest_indices = np.argmin(distances, axis=1)
		assignments = ref_vectors[nearest_indices]

		return np.expand_dims(nearest_indices, -1)
	
	ref_idx = assign_ref(ref_dirs, pareto_y_task)
	
	parents = []
	
	for i in range(int(raw_samples/2)):
		rand_idx = np.random.choice(batch_size)
		pool_a = pareto_x_task[(ref_idx==rand_idx).all(axis=1)]
		while pool_a.shape[0] == 0:
			rand_idx = np.random.randint(batch_size)
			pool_a = pareto_x_task[(ref_idx==rand_idx).all(axis=1)]
		pool_b = pareto_x_task[(ref_idx!=rand_idx).all(axis=1)]
		if pool_b.shape[0] == 0:
			pool_b = pool_a
	
		a = Individual(X=pool_a[np.random.choice(pool_a.shape[0])])
		b = Individual(X=pool_b[np.random.choice(pool_b.shape[0])])
		parents.append([a,b])
	
	xl = np.zeros(x.shape[1]+1)
	xu = np.ones(x.shape[1]+1)
	xu[-1] = int(task.max())
	pymooproblem = PymooProblem(n_var=x.shape[1]+1, xl=xl, xu=xu)
	off = SBX(prob=1.0, prob_var=1.0).do(pymooproblem, parents)
	off = PolynomialMutation(prob=1.0)(pymooproblem, off)
	
	# assign task indicator column to inherit either of parents
	off_task = []
	for i in range(len(parents)):
		parent_task = np.array([parents[i][0].get('X')[-1], parents[i][1].get('X')[-1]])
		off_task.append(np.random.choice(parent_task, 2).reshape(-1))
	
	off_x = off.get('X')[:,:-1] # remove old task indicator
	off_task = np.array(off_task).reshape(-1,1)
	candidates = torch.tensor(np.hstack([off_x, off_task]), **tkwargs)
	
	acq_value_list = [acq_func(candidates[i].unsqueeze(dim=0)).detach().item()
					  for i in range(candidates.shape[0])]
	sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]
	
	return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples

def optimize_mt_egbo_list(acq_func, x, task, y, batch_size):
	n_tasks = int(task.max()+1)
	
	pareto_mask = is_non_dominated(y[(task==0).all(dim=1)])
	pareto_x = x[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	pareto_task = task[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	pareto_y = y[(task==0).all(dim=1)][pareto_mask].cpu().numpy()
	
	for i in range(1, int(task.max())+1):
		pareto_mask = is_non_dominated(y[(task==i).all(dim=1)])
		pareto_x = np.vstack([pareto_x, x[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
		pareto_task = np.vstack([pareto_task, task[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
		pareto_y = np.vstack([pareto_y, y[(task==i).all(dim=1)][pareto_mask].cpu().numpy()])
	
	
	ref_dirs = get_reference_directions("energy", y.shape[1]+1, n_tasks)
	for i in range(y.shape[1]):
		if y[0,i] < 0: # ensure correct direction if we have minimization objectives
			ref_dirs[:,i] = -ref_dirs[:,i]
	
	# we select for pareto solutions which are spread in both y and task space
	scaler = MinMaxScaler()
	pareto_y_task = scaler.fit_transform(np.hstack([pareto_y, pareto_task]))
	pareto_x_task = np.hstack([pareto_x, pareto_task])
	
	def assign_ref(ref_vectors, points):
		distances = np.linalg.norm(points[:, np.newaxis] - ref_vectors, axis=2)
		nearest_indices = np.argmin(distances, axis=1)
		assignments = ref_vectors[nearest_indices]

		return np.expand_dims(nearest_indices, -1)
	
	ref_idx = assign_ref(ref_dirs, pareto_y_task)
	
	parents = []
	
	for i in range(int(raw_samples/2)):
		rand_idx = np.random.choice(batch_size)
		pool_a = pareto_x_task[(ref_idx==rand_idx).all(axis=1)]
		while pool_a.shape[0] == 0:
			rand_idx = np.random.randint(batch_size)
			pool_a = pareto_x_task[(ref_idx==rand_idx).all(axis=1)]
		pool_b = pareto_x_task[(ref_idx!=rand_idx).all(axis=1)]
		if pool_b.shape[0] == 0:
			pool_b = pool_a
	
		a = Individual(X=pool_a[np.random.choice(pool_a.shape[0])])
		b = Individual(X=pool_b[np.random.choice(pool_b.shape[0])])
		parents.append([a,b])
	
	xl = np.zeros(x.shape[1]+1)
	xu = np.ones(x.shape[1]+1)
	xu[-1] = int(task.max())
	pymooproblem = PymooProblem(n_var=x.shape[1]+1, xl=xl, xu=xu)
	off = SBX(prob=1.0, prob_var=1.0).do(pymooproblem, parents)
	off = PolynomialMutation(prob=1.0)(pymooproblem, off)
	
	# assign task indicator column to inherit either of parents
	off_task = []
	for i in range(len(parents)):
		parent_task = np.array([parents[i][0].get('X')[-1], parents[i][1].get('X')[-1]])
		off_task.append(np.random.choice(parent_task, 2).reshape(-1))
	
	off_x = off.get('X')[:,:-1] # remove old task indicator
	off_task = np.array(off_task).reshape(-1,1)
	candidates = torch.tensor(np.hstack([off_x, off_task]), **tkwargs)
	
	acq_value_list = [acq_func[np.random.choice(len(acq_func))](candidates[i].unsqueeze(dim=0)).detach().item() for i in range(candidates.shape[0])]
	sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]
	
	return torch.tensor(sorted_x[-batch_size:], **tkwargs) # take best BATCH_SIZE samples
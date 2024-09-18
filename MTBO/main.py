import torch
import math
import time
import numpy as np

from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch import fit_gpytorch_mll

from MTBO.sampling import LHS
from MTBO.utils import calc_hv, update_values
from MTBO.models import initialize_model_st, initialize_model_mt
from MTBO.acq_func import mt_qnehvi, mt_qnparego, mt_qucb, st_qnehvi, st_qnparego, st_qucb
from MTBO.optim import optimize_mt_list, optimize_mt_mixed, optimize_st_list, optimize_st_acqf, optimize_st_egbo, optimize_mt_egbo

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}


class MTBO():

	def __init__(self, problems):
        
		if len(problems) < 2:
			raise Exception("Need at least 2 problems")
		
		self.problems = problems
		self.prob_bounds = problems[0].bounds
		
		self.acq_bounds = torch.zeros((2, problems[0].n_var+1), **tkwargs)
		self.acq_bounds[1] = 1
		self.acq_bounds[1,-1] = len(problems)-1
		
		self.std_bounds = torch.zeros((2, problems[0].n_var), **tkwargs)
		self.std_bounds[1] = 1

		self.ref_pt = problems[0].ref_pt
		self.hv = Hypervolume(ref_point=problems[0].ref_pt)
		self.n_task = len(problems)
		self.n_obj = problems[0].n_obj
		self.n_var = problems[0].n_var

	def methods(self):
		print(
		f"Multi-Task: MT-qNEHVI, MT-qNEHVI(EGBO), MT-qNParEGO\n"
		f"Single-Task: ST-qNEHVI, ST-qNEHVI(EGBO), ST-qNParEGO, ST-qUCB"
		)

	def initialize(self, n_init, random_state=np.random.randint(99999)):
		self.init_x, self.init_task, self.init_y = LHS(self.problems, n_init)

	def run(self, n_iter, n_batch, task_type, algo, random_state=np.random.randint(99999)):
		print(f"Optimizing for {task_type}-{algo}")

		torch.manual_seed(random_state)
		np.random.seed(random_state)
		
		results  = []
		
		#### initialization ####
		
		train_x, train_task, train_y = self.init_x, self.init_task, self.init_y
		volumes = calc_hv(train_y, train_task, self.hv, self.problems)
		results.append(volumes)
		
		if task_type == 'multi':
			x_gp = normalize(train_x, self.prob_bounds)   
			model, mll = initialize_model_mt(x_gp, train_task, train_y)
		
		elif task_type == 'single':
			x_gp = [
				normalize(train_x[(train_task==i).all(dim=1)],
				self.prob_bounds) for i in range(self.n_task)
			]
			model_list, mll_list = [], []
			for i in range(self.n_task):
				model_temp, mll_temp = initialize_model_st(x_gp[i], train_y[(train_task==i).all(dim=1)])
				model_list.append(model_temp)
				mll_list.append(mll_temp)
			
		print(f"Batch 0 - avg HV:{volumes.mean():.4f}")
		
		for iter in range(1, n_iter+1):
			t2 = time.monotonic()
			
			if task_type == 'multi':
				fit_gpytorch_mll(mll)
				
				if algo == 'qnehvi':
					acq = mt_qnehvi(model, self.ref_pt, x_gp, train_task)
					candidates =  optimize_mt_mixed(acq, self.acq_bounds, n_batch*self.n_task)
				elif algo == 'egbo':
					acq = mt_qnehvi(model, self.ref_pt, x_gp, train_task)
					candidates = optimize_mt_egbo(acq, self.ref_pt, x_gp, train_task, train_y, n_batch*self.n_task)
				elif algo == 'qnparego':
					acq = mt_qnparego(model, x_gp, train_task, n_batch*self.n_task, self.n_obj)
					candidates = optimize_mt_list(acq, self.acq_bounds)
				elif algo == 'qucb':
					acq = mt_qucb(model, x_gp, train_task, n_batch*self.n_task, self.n_obj)
					candidates = optimize_mt_list(acq, self.acq_bounds)
				
				new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
				new_task = candidates[:,-1].unsqueeze(1)
				train_x, train_task, train_y = update_values(
					(train_x, train_task, train_y), (new_x, new_task), self.problems)
		
				x_gp = normalize(train_x, self.prob_bounds)   
				model, mll = initialize_model_mt(x_gp, train_task, train_y)
			
			elif task_type == 'single':
				for i in range(self.n_task):
					fit_gpytorch_mll(mll_list[i])
					new_x = []
		
					for i in range(self.n_task):
						if algo == 'qnehvi':
							acq = st_qnehvi(model_list[i], self.ref_pt, x_gp[i])
							candidates = optimize_st_acqf(acq, n_batch, self.std_bounds)
						elif algo == 'egbo':
							acq = st_qnehvi(model_list[i], self.ref_pt, x_gp[i])
							candidates = optimize_st_egbo(acq, self.ref_pt, x_gp[i],
													   train_y[(train_task==i).all(dim=1)], n_batch)
						elif algo == 'qnparego':
							acq = st_qnparego(model_list[i], x_gp[i], n_batch, self.n_obj)
							candidates = optimize_st_list(acq, self.std_bounds)
						elif algo == 'qucb':
							acq = st_qucb(model_list[i], x_gp[i], n_batch, self.n_obj)
							candidates = optimize_st_list(acq, self.std_bounds)
		
						new_x.append(unnormalize(candidates, self.prob_bounds).cpu().numpy())
		
					new_x = torch.tensor(np.array(new_x), **tkwargs).reshape(n_batch*self.n_task, self.n_var)
					new_task = torch.tensor([
						task for task in range(self.n_task)], **tkwargs).tile(n_batch).unsqueeze(1)
			
				train_x, train_task, train_y = update_values(
					(train_x, train_task, train_y), (new_x, new_task), self.problems)
		
				x_gp = [
					normalize(train_x[(train_task==i).all(dim=1)], self.prob_bounds) for i in range(self.n_task)]
				model_list, mll_list = [], []
				for i in range(self.n_task):
					model_temp, mll_temp = initialize_model_st(x_gp[i], train_y[(train_task==i).all(dim=1)])
					model_list.append(model_temp)
					mll_list.append(mll_temp)
		
			####
			
			volumes = calc_hv(train_y, train_task, self.hv, self.problems)
			results.append(volumes)
			
			t3 = time.monotonic()
			print(f"Batch {iter} - avg HV:{volumes.mean():.4f}, time:{t3-t2:>4.2f}")
			torch.cuda.empty_cache()
		
		return np.array(results)
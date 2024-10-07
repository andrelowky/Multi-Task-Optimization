import torch
import math
import time
import numpy as np

from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch import fit_gpytorch_mll

from MTBO.sampling import sobol_repeated, LHS, sobol
from MTBO.utils import calc_hv, update_values, calc_losses
from MTBO.models import initialize_model_st, initialize_model_mt, initialize_fit_model_ftgp
from MTBO.acq_func import st_qnehvi, st_qnparego, st_qucb, mt_qnehvi, mt_qnparego, mt_qucb
from MTBO.optim import optimize_list, optimize_fixed_list, optimize_mt_mixed, optimize_st_acqf, optimize_st_egbo, optimize_mt_egbo

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

	def initialize(self, n_init, sampling='sobol_repeated', random_state=np.random.randint(99999)):
		if sampling == 'sobol_repeated':
			self.init_x, self.init_task, self.init_y = sobol_repeated(self.problems, n_init, random_state)
		else:
			self.init_x, self.init_task, self.init_y = LHS(self.problems, n_init, random_state)

	def run(self, n_iter, n_batch, task_type, algo, model_type='standard', random_state=np.random.randint(99999)):
		print(f"Optimizing for {task_type}-{algo}-{model_type}")

		torch.manual_seed(random_state)
		np.random.seed(random_state)
		
		results  = []
		
		#### initialization ####
		
		self.train_x, self.train_task, self.train_y = self.init_x, self.init_task, self.init_y
		x_gp = normalize(self.train_x[(self.train_task==0).all(dim=1)], self.prob_bounds)   
		for i in range(1, self.n_task):
			x_gp = torch.vstack([x_gp, normalize(self.train_x[(self.train_task==i).all(dim=1)], self.prob_bounds)])
		volumes = calc_hv(self.train_y, self.train_task, self.hv, self.problems)
		results.append(volumes)
		
		print(f"Batch 0 - avg HV:{volumes.mean():.4f}")
		
		for iter in range(1, n_iter+1):
			t2 = time.monotonic()
		
			if task_type == 'multi':

				if model_type=='standard':
				
					model, mll = initialize_model_mt(x_gp, self.train_task, self.train_y)
					fit_gpytorch_mll(mll)
					'''
					optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
					n_epoch = 100
					
					for i in range(n_epoch):
						optimizer.zero_grad()
						output = model(*model.train_inputs)
						loss = -mll(output, model.train_targets)
						loss.backward()
						optimizer.step()
					'''

				else:
					model = initialize_fit_model_ftgp(x_gp, self.train_task, self.train_y)

				if algo == 'qnehvi':
					acq = mt_qnehvi(model, self.ref_pt, x_gp, self.train_task)
					candidates = optimize_mt_mixed(acq, self.acq_bounds, self.n_task*n_batch, self.n_task)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
					
				elif algo == 'egbo':
					acq = mt_qnehvi(model, self.ref_pt, x_gp, self.train_task)
					candidates = optimize_mt_egbo(acq, x_gp, self.train_task, self.train_y, n_batch, self.n_obj)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
					
				elif algo == 'qnparego':
					acq = mt_qnparego(model, x_gp, self.train_task, n_batch, self.n_obj)
					candidates = optimize_fixed_list(acq, self.acq_bounds, self.n_task)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
					
				elif algo == 'qucb':
					acq = mt_qucb(model, x_gp, self.train_task, n_batch, self.n_obj)
					candidates = optimize_fixed_list(acq, self.acq_bounds, self.n_task)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
			
			elif task_type == 'single':

				new_x = []
				for i in range(self.n_task):
	
					model, mll = initialize_model_st(
						x_gp[(self.train_task==i).all(dim=1)],
						self.train_y[(self.train_task==i).all(dim=1)])
					fit_gpytorch_mll(mll)

					if algo == 'qnehvi':
						acq = st_qnehvi(model, self.ref_pt, x_gp)
						candidates = optimize_st_acqf(acq, n_batch, self.std_bounds)
					elif algo == 'egbo':
						acq = st_qnehvi(model, self.ref_pt, x_gp)
						candidates = optimize_st_egbo(acq, x_gp, self.train_y[(self.train_task==i).all(dim=1)], n_batch)
					elif algo == 'qnparego':
						acq = st_qnparego(model, x_gp, n_batch, self.n_obj)
						candidates = optimize_list(acq, self.std_bounds)
					elif algo == 'qucb':
						acq = st_qucb(model, x_gp, n_batch, self.n_obj)
						candidates = optimize_list(acq, self.std_bounds)
	
					new_x.append(unnormalize(candidates, self.prob_bounds).cpu().numpy())
			
				new_x = torch.tensor(np.array(new_x), **tkwargs).reshape(n_batch*self.n_task, self.n_var)
				new_task = torch.tensor([task for task in range(self.n_task)], **tkwargs).tile(n_batch).unsqueeze(1)


			#### update and go next iteration
			self.train_x, self.train_task, self.train_y = update_values(
				(self.train_x, self.train_task, self.train_y), 
				(new_x, new_task), self.problems)

			x_gp = normalize(self.train_x[(self.train_task==0).all(dim=1)], self.prob_bounds)   
			for i in range(1, self.n_task):
				x_gp = torch.vstack([x_gp, normalize(self.train_x[(self.train_task==i).all(dim=1)], self.prob_bounds)])
			
			volumes = calc_hv(self.train_y, self.train_task, self.hv, self.problems)
			results.append(volumes)
			
			t3 = time.monotonic()
			print(f"Batch {iter} - avg HV:{volumes.mean():.4f}, time:{t3-t2:>4.2f}")
			
			del model, acq, new_x, new_task
			torch.cuda.empty_cache()

		return np.array(results)
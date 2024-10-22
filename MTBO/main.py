import torch
import math
import time
import numpy as np

from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch import fit_gpytorch_mll

from MTBO.sampling import anchored_sampling
from MTBO.utils import calc_hv, update_values, calc_losses
from MTBO.models import initialize_model_st, initialize_model_mt, initialize_fit_model_ftgp
from MTBO.acq_func import st_qnehvi, st_qnparego, st_qucb, mt_qnehvi, mt_qnparego, mt_qucb
from MTBO.optim import optimize_st_list, optimize_mt_list_flexi, optimize_mt_list_fixed, optimize_mt_mixed_flexi, optimize_mt_mixed_fixed, optimize_st_acqf, optimize_st_egbo, optimize_mt_egbo, optimize_mt_egbo_list

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

	def initialize(self, n_init, random_state=np.random.randint(99999)):
		# split half for repeated, and half for coverage
		
		self.init_x, self.init_task, self.init_y = anchored_sampling(self.problems, n_init, random_state)

	def run(self, n_iter, n_batch, 
			task_type, algo, model_type, final_batch=False,
			random_state=np.random.randint(99999)):
		print(f"Optimizing for {task_type}-{algo}-{model_type}")

		torch.manual_seed(random_state)
		np.random.seed(random_state)
		
		results  = []
		
		#### initialization ####
		
		self.train_x, self.train_task, self.train_y = self.init_x, self.init_task, self.init_y
		x_gp = normalize(self.train_x, self.prob_bounds)   
		'''
		x_gp = normalize(self.train_x[(self.train_task==0).all(dim=1)], self.prob_bounds)   
		for i in range(1, self.n_task):
			x_gp = torch.vstack([x_gp, normalize(self.train_x[(self.train_task==i).all(dim=1)], self.prob_bounds)])
		'''
		volumes = calc_hv(self.train_y, self.train_task, self.hv, self.problems)
		results.append(volumes)
		
		print(f"Batch 0 - avg HV:{volumes.mean():.4f}")
		
		for iter in range(1, n_iter+1):
			t2 = time.monotonic()
		
			if task_type == 'multi-flexi':

				if model_type=='mtgp':
				
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
					candidates = optimize_mt_mixed_flexi(acq, self.acq_bounds, self.n_task*n_batch, self.n_task)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)

				elif algo == 'qnehvi-egbo':
					acq = mt_qnehvi(model, self.ref_pt, x_gp, self.train_task)
					candidates = optimize_mt_egbo(acq, x_gp, self.train_task, self.train_y, self.n_task*n_batch)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
					
				elif algo == 'qnparego':
					acq = mt_qnparego(model, x_gp, self.train_task, self.n_task*n_batch, self.n_obj)
					candidates = optimize_mt_list_flexi(acq, self.acq_bounds, self.n_task)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)

				elif algo == 'qnparego-egbo':
					acq = mt_qnparego(model, x_gp, self.train_task, self.n_task, self.n_obj)
					candidates = optimize_mt_egbo_list(acq, x_gp, self.train_task, self.train_y, self.n_task*n_batch)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
					
				elif algo == 'qucb':
					acq = mt_qucb(model, x_gp, self.train_task, self.n_task*n_batch, self.n_obj)
					candidates = optimize_mt_list_flexi(acq, self.acq_bounds, self.n_task)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)

				elif algo == 'qucb-egbo':
					acq = mt_qucb(model, x_gp, self.train_task, self.n_task, self.n_obj)
					candidates = optimize_mt_egbo_list(acq, x_gp, self.train_task, self.train_y, self.n_task*n_batch)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)

			elif task_type == 'multi-fixed':

				if model_type=='standard':
				
					model, mll = initialize_model_mt(x_gp, self.train_task, self.train_y)
					fit_gpytorch_mll(mll)

				else:
					model = initialize_fit_model_ftgp(x_gp, self.train_task, self.train_y)

				if algo == 'qnehvi':
					acq = mt_qnehvi(model, self.ref_pt, x_gp, self.train_task)
					candidates = []
					for task_idx in range(self.n_task):
						candidates.append(optimize_mt_mixed_fixed(acq, self.acq_bounds, n_batch, task_idx).cpu().numpy())
					candidates = torch.tensor(np.array(candidates), **tkwargs).reshape(-1, self.n_var+1)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)

				elif algo == 'qnehvi-egbo':
					acq = mt_qnehvi(model, self.ref_pt, x_gp, self.train_task)
					candidates = []
					for task_idx in range(self.n_task):
						x_gp_i = x_gp[(self.train_task==task_idx).all(dim=1)]
						train_task_i = self.train_task[(self.train_task==task_idx).all(dim=1)]
						train_y_i = self.train_y[(self.train_task==task_idx).all(dim=1)]
						candidates_i = optimize_st_egbo(acq, x_gp_i, train_y_i, n_batch)
						candidates_i = torch.hstack([candidates_i, torch.tensor(task_idx).repeat(n_batch).unsqueeze(1)])
						candidates.append(candidates_i.cpu().numpy())

					candidates = torch.tensor(np.array(candidates), **tkwargs).reshape(-1, self.n_var+1)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)

				elif algo == 'qnparego':
					acq = mt_qnparego(model, x_gp, self.train_task, n_batch, self.n_obj)
					candidates = []
					for task_idx in range(self.n_task):
						candidates.append(optimize_mt_list_fixed(acq, self.acq_bounds, task_idx).cpu().numpy())
					candidates = torch.tensor(np.array(candidates), **tkwargs).reshape(-1, self.n_var+1)
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
					
				elif algo == 'qucb':
					acq = mt_qucb(model, x_gp, self.train_task, n_batch, self.n_obj)
					candidates = []
					for task_idx in range(self.n_task):
						candidates.append(optimize_mt_list_fixed(acq, self.acq_bounds, task_idx).cpu().numpy())
					candidates = torch.tensor(np.array(candidates), **tkwargs).reshape(-1, self.n_var+1)				
					new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
					new_task = candidates[:,-1].unsqueeze(1)
			
			elif task_type == 'single':

				new_x = []
				for i in range(self.n_task):

					x_gp_i = x_gp[(self.train_task==i).all(dim=1)]
					train_y_i = self.train_y[(self.train_task==i).all(dim=1)]
	
					model, mll = initialize_model_st(x_gp_i, train_y_i)
					fit_gpytorch_mll(mll)

					if algo == 'qnehvi':
						acq = st_qnehvi(model, self.ref_pt, x_gp_i)
						candidates = optimize_st_acqf(acq, n_batch, self.std_bounds)
					
					elif algo == 'qnehvi-egbo':
						acq = st_qnehvi(model, self.ref_pt, x_gp_i)
						candidates = optimize_st_egbo(acq, x_gp_i, train_y_i, n_batch)
					
					elif algo == 'qnparego':
						acq = st_qnparego(model, x_gp_i, n_batch, self.n_obj)
						candidates = optimize_st_list(acq, self.std_bounds)
					elif algo == 'qucb':
						acq = st_qucb(model, x_gp_i, n_batch, self.n_obj)
						candidates = optimize_st_list(acq, self.std_bounds)
	
					new_x.append(unnormalize(candidates, self.prob_bounds).cpu().numpy())
			
				new_x = torch.tensor(np.array(new_x), **tkwargs).reshape(-1, self.n_var)
				new_task = torch.tensor([np.array(task).repeat(n_batch) for task in range(self.n_task)], **tkwargs).reshape(-1).unsqueeze(1)

			#### update and go next iteration
			self.train_x, self.train_task, self.train_y = update_values(
				(self.train_x, self.train_task, self.train_y), 
				(new_x, new_task), self.problems)

			x_gp = normalize(self.train_x, self.prob_bounds)   

			volumes = calc_hv(self.train_y, self.train_task, self.hv, self.problems)
			results.append(volumes)
			
			t3 = time.monotonic()
			print(f"Batch {iter} - avg HV:{volumes.mean():.4f}, time:{t3-t2:>4.2f}")
			
			del model, acq, new_x, new_task
			torch.cuda.empty_cache()


		#### the final validation batch
		if final_batch:
			t2 = time.monotonic()
			if task_type == 'multi-fixed' or task_type == 'multi-flexi':
				model, mll = initialize_model_mt(x_gp, self.train_task, self.train_y)
				fit_gpytorch_mll(mll)
				acq = mt_qnehvi(model, self.ref_pt, x_gp, self.train_task)
				candidates = []
				for task_idx in range(self.n_task):
					
					candidates.append(optimize_mt_mixed_fixed(acq, self.acq_bounds, n_batch, task_idx).cpu().numpy())
				candidates = torch.tensor(np.array(candidates), **tkwargs).reshape(-1, self.n_var+1)
				new_x = unnormalize(candidates[:,:-1], self.prob_bounds)
				new_task = candidates[:,-1].unsqueeze(1)
	
			elif task_type == 'single':
				new_x = []
				for i in range(self.n_task):
					print(f"Performing final validation for task {i+1}")
					x_gp_i = x_gp[(self.train_task==i).all(dim=1)]
					train_y_i = self.train_y[(self.train_task==i).all(dim=1)]
	
					model, mll = initialize_model_st(x_gp_i, train_y_i)
					fit_gpytorch_mll(mll)
					acq = st_qnehvi(model, self.ref_pt, x_gp_i)
					candidates = optimize_st_acqf(acq, n_batch, self.std_bounds)	
				
					new_x.append(unnormalize(candidates, self.prob_bounds).cpu().numpy())
				
				new_x = torch.tensor(np.array(new_x), **tkwargs).reshape(-1, self.n_var)
				new_task = torch.tensor([np.array(task).repeat(n_batch) for task in range(self.n_task)], **tkwargs).reshape(-1).unsqueeze(1)
				
			self.train_x, self.train_task, self.train_y = update_values(
				(self.train_x, self.train_task, self.train_y), 
				(new_x, new_task), self.problems)
	
			x_gp = normalize(self.train_x, self.prob_bounds)   
	
			volumes = calc_hv(self.train_y, self.train_task, self.hv, self.problems)
			results.append(volumes)
			
			t3 = time.monotonic()
			print(f"Final batch - avg HV:{volumes.mean():.4f}, time:{t3-t2:>4.2f}")
		
		return np.array(results), (self.train_x.cpu().numpy(), self.train_task.cpu().numpy(), self.train_y.cpu().numpy())
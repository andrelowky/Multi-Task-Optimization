import os
import torch
import math
import time
import numpy as np
import argparse
import joblib
from datetime import datetime

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from MTBO.main import MTBO
from MTBO.problems import DTLZ1, DTLZ2, ZDT1

main_dir = os.getcwd()

def main(args):

	problem_list = []
	'''
	for i in range(2):
		problem_list.append(DTLZ1(n_var=args.n_var,
								  delta1 = 1, delta2 = 0.1*torch.rand(1).item()-0.05, delta3 = 1,
								  negate=True))
	
	for i in range(2):
		problem_list.append(DTLZ1(n_var=args.n_var,
								  delta1 = 1, delta2 = 0.1*torch.rand(1).item()+0.1, delta3 = 1,
								  negate=True))

	for i in range(2):
		problem_list.append(DTLZ1(n_var=args.n_var,
								  delta1 = 1, delta2 = 0.1*torch.rand(1).item()+0.2, delta3 = 1,
								  negate=True))
	'''	
	problem_list.append(ZDT1(n_var=args.n_var, delta1=0, negate=True))
	problem_list.append(ZDT1(n_var=args.n_var, delta1=0.1, negate=True))
	problem_list.append(ZDT1(n_var=args.n_var, delta1=0.2, negate=True))
	
	opt = MTBO(problem_list)

	all_results = {}
	all_data = {}
	all_losses = {}

	taskalgo_list = [
		('multi-flexi', 'qnehvi'),
		('multi-flexi', 'qnehvi-egbo'),
		('multi-flexi', 'qnehvi-sobol'),
		('multi-fixed', 'qnehvi'),
		('multi-fixed', 'qnehvi-egbo'),
		#('multi-fixed', 'qucb'),
		#('multi-fixed', 'qnehvi-egbo',
		#('multi-fixed', 'qucb-egbo'),
		('single', 'qnehvi'),
		('single', 'random'),
	]

	for i, taskalgo in enumerate(taskalgo_list):
		task_type, algo = taskalgo
		
		all_results[f"{task_type}-{algo}"] = []
		all_data[f"{task_type}-{algo}"] = []
		all_losses[f"{task_type}-{algo}"] = []
	
	for trial in range(1, args.n_trial+1):
		print(f"Trial {trial}/{args.n_trial}")
		opt.initialize(n_init=args.n_init, random_state=trial)
		
		for i, taskalgo in enumerate(taskalgo_list):
			task_type, algo = taskalgo
			opt.run(
				n_iter=args.n_iter, n_batch=args.n_batch,
				task_type=task_type, algo=algo)
			
			#opt.validate(n_batch_final=args.n_batch_final)
			results, data, losses = opt.output_results()
			opt.reset()
			
			all_results[f"{task_type}-{algo}"].append(results)
			all_data[f"{task_type}-{algo}"].append(data)
			all_losses[f"{task_type}-{algo}"].append(losses)
			
	joblib.dump(all_results, f'{main_dir}/results/{args.problem_main}-results-{args.label}')
	joblib.dump(all_data, f'{main_dir}/results/{args.problem_main}-data-{args.label}')
	joblib.dump(all_losses, f'{main_dir}/results/{args.problem_main}-losses-{args.label}')
	
	print("Done!")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--problem_main', default='ZDT1', type=str)
	parser.add_argument('--n_var', default=4, type=int)

	parser.add_argument('--n_trial', default=5, type=int)
	parser.add_argument('--n_iter', default=2, type=int)
	parser.add_argument('--n_batch', default=6, type=int)
	parser.add_argument('--n_init', default=12, type=int)
	parser.add_argument('--n_batch_final', default=2, type=int)
		
	parser.add_argument('--label', default='', type=str)
	
	args = parser.parse_args()
	main(args)


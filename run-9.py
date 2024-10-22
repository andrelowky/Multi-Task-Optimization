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
from MTBO.problems import DTLZ1, DTLZ2


def main(args):

	problem_list = []
	for i in range(4):
		problem_list.append(DTLZ1(n_var=args.n_var,
								  delta1 = 1, delta2 = 0.1*torch.rand(1).item()-0.05, delta3 = 1,
								  negate=True))
	for i in range(4):
		problem_list.append(DTLZ1(n_var=args.n_var,
								  delta1 = 1, delta2 = 0.1*torch.rand(1).item()+0.2, delta3 = 1,
								  negate=True))

	problem_list.append(DTLZ1(n_var=args.n_var,
							  delta1 = 1, delta2 = 0.1*torch.rand(1).item()+0.1, delta3 = 1,
							  negate=True))
		
	opt = MTBO(problem_list)

	all_results = {f'results{i}': [] for i in range(0, 7)}
	all_data = {f'data{i}': [] for i in range(0, 7)}
	
	for trial in range(1, args.n_trial+1):
		print(f"Trial {trial}/{args.n_trial}")
		opt.initialize(n_init=args.n_init, random_state=trial)
		
		# Create trial-specific results and data lists
		trial_results = []
		trial_data = []
		
		for i, taskalgo in enumerate([
			#('multi-flexi', 'qnehvi-egbo', 'ftgp'),
			#('multi-flexi', 'qucb-egbo', 'ftgp'),
			('multi-flexi', 'qnehvi-egbo', 'mtgp'),
			('multi-flexi', 'qucb-egbo', 'mtgp'),
			('single', 'qnehvi-egbo', 'stgp')
		]):
			task_type, algo, model_type = taskalgo
			results, data = opt.run(
				n_iter=args.n_iter, n_batch=args.n_batch,
				task_type=task_type, algo=algo, model_type=model_type, 
				final_batch=args.final_batch,
				random_state=trial)
			all_results[f'results{i}'].append(results)
			all_data[f'data{i}'].append(data)
		

	for i, taskalgo in enumerate([
		#('multi-flexi', 'qnehvi-egbo', 'ftgp'),
		#('multi-flexi', 'qucb-egbo', 'ftgp'),
		('multi-flexi', 'qnehvi-egbo', 'mtgp'),
		('multi-flexi', 'qucb-egbo', 'mtgp'),
		('single', 'qnehvi-egbo', 'stgp')
	]):
		task_type, algo, model_type = taskalgo
		results_list = np.array(all_results[f'results{i}'])
		data_list = all_data[f'data{i}']
		joblib.dump(results_list, f'{args.problem_main}-results-{task_type}-{algo}-{model_type}-{args.label}')
		joblib.dump(data_list, f'{args.problem_main}-data-{task_type}-{algo}-{model_type}-{args.label}')
	
	print("Done!")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--problem_main', default='DTLZ1', type=str)
	parser.add_argument('--n_var', default=4, type=int)

	parser.add_argument('--n_trial', default=5, type=int)
	parser.add_argument('--n_iter', default=6, type=int)
	parser.add_argument('--n_batch', default=2, type=int)
	parser.add_argument('--n_init', default=36, type=int)
	
	parser.add_argument('--final_batch', default=False, type=bool)
	
	parser.add_argument('--label', default='', type=str)
	
	args = parser.parse_args()
	main(args)


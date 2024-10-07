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

accepted_problems = ['DTLZ1', 'DTLZ2']
accepted_task_types = ['multi', 'single']
accepted_algos = ['qnehvi', 'egbo', 'qnparego', 'qucb']

def main(args):

	if args.problem_main not in accepted_problems:
	    raise ValueError(f"Problem must be in {accepted_problems}")
	if args.task_type not in accepted_task_types:
	    raise ValueError(f"Task type must be in {task_types}")
	if args.algo not in accepted_algos:
	    raise ValueError(f"Algo must be in {accepted_algos}")

	if args.problem_main == 'DTLZ1':
		problem_main = DTLZ1
	elif args.problem_main == 'DTLZ2':
		problem_main = DTLZ2

	problem_list = []
	delta2 = 0
	for i in range(args.n_problem):
		problem_list.append(DTLZ1(n_var=args.n_var,
								  delta1 = 1, delta2 = delta2, delta3 = 1,
								  negate=True))
		delta2+=args.corr
		
	opt = MTBO(problem_list)

	if args.run_all: # everything

		results1, results2, results3, results4, results5, results6 = [], [], [], [], [], []
		
		for trial in range(1, args.n_trial+1):
			
			print(f"Trial {trial}/{args.n_trial}")
			opt.initialize(n_init=args.n_init, random_state=trial)

			for taskalgo, results_list in zip(
				[('multi', 'qnparego', 'standard'), ('multi','qucb','standard'), 
				 ('multi', 'qnparego', 'ftgp'), ('multi','qucb','ftgp'), 
				 ('single', 'qnparego','standard'), ('single','qucb','standard'),],
				[results1, results2, results3, results4, results5, results6],
			):

				task_type, algo, model_type = taskalgo
				results = opt.run(n_iter=args.n_iter, n_batch=args.n_batch,
								  task_type=task_type, algo=algo, model_type=model_type, random_state=trial)
				results_list.append(results)
			

		for taskalgo, results_list in zip(
			[('multi', 'qnparego', 'standard'), ('multi','qucb','standard'), 
			 ('multi', 'qnparego','ftgp'), ('multi','ftgp','qucb','ftgp'), 
			 ('single', 'qnparego','standard'), ('single','qucb','standard'),],
			[results1, results2, results3, results4, results5, results6],
		):
			task_type, algo, model_type = taskalgo
			results_list = np.array(results_list)
			joblib.dump(results_list, f'{args.problem_main}-results-{task_type}-{algo}-{model_type}-{args.label}')
	
	else: # single runs
		results_all = []
		for trial in range(1, args.n_trial+1):
			print(f"Trial {trial}/{args.n_trial}")

			opt.initialize(n_init=args.n_init, random_state=trial)
			results = opt.run(n_iter=args.n_iter, n_batch=args.n_batch,
							  task_type=args.task_type, algo=args.algo, model_type=args.model_type, random_state=trial)
			results_all.append(results)
	
		results_all = np.array(results_all)
	
		joblib.dump(results_all, f'{args.problem_main}-results-{args.task_type}-{args.algo}-{args.model_type}-{args.label}')
	
	
	print("Done!")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--problem_main', default='DTLZ1', type=str)
	parser.add_argument('--n_problem', default=3, type=int)
	parser.add_argument('--corr', default=0.05, type=int)
	parser.add_argument('--n_var', default=6, type=int)

	parser.add_argument('--n_trial', default=5, type=int)
	parser.add_argument('--n_iter', default=4, type=int)
	parser.add_argument('--n_batch', default=2, type=int)
	parser.add_argument('--n_init', default=36, type=int)
	
	parser.add_argument('--task_type', default='multi', type=str)
	parser.add_argument('--algo', default='qnehvi', type=str)
	parser.add_argument('--model_type', default='standard', type=str)
	parser.add_argument('--label', default='', type=str)

	parser.add_argument('--run_all', default=False)
	
	args = parser.parse_args()
	main(args)


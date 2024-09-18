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
from MTBO.problems import DTLZ1

accepted_problems = ['DTLZ1']
accepted_task_types = ['multi', 'single']
accepted_algos = ['qnehvi', 'egbo', 'qnparego', 'qucb']

def main(args):

	if args.problem_main not in accepted_problems:
	    raise ValueError(f"Problem must be in {accepted_problems}")
	if args.task_type not in accepted_task_types:
	    raise ValueError(f"Task type must be in {task_types}")
	if args.algo not in accepted_algos:
	    raise ValueError(f"Algo must be in {accepted_algos}")
	if args.problem_main == 'qucb' and args.task_type == 'multi':
		raise ValueError(f"Multi-task qUCB not supported")
	

	if args.problem_main == 'DTLZ1':
		problem_main = DTLZ1

	problem_list = []
	corr = 0
	for i in range(args.n_problems):
		problem_list.append(DTLZ1(n_var=6, delta1 = 1, delta2 = corr, delta3 = 1, negate=True))
		corr+=0.1
	
	opt = MTBO(problem_list)

	results_all = []
	for trial in range(1, args.n_trial+1):
		print(f"Trial {trial}/{args.n_trial}")
		
		results = opt.run(n_iter=args.n_iter, n_batch=args.n_batch, n_init=args.n_init,
						  task_type=args.task_type, algo=args.algo, random_state=trial)
		results_all.append(results)

	results_all = np.array(results_all)

	if args.label == '':
		today = datetime.now().strftime("%d-%m-%y--%H:%M")
		joblib.dump(results_all, f'{today}')

	else:
		joblib.dump(results_all, f'{args.label}')

	print("Done!")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--problem_main', required=True)
	parser.add_argument('--n_problems', required=True, type=int)

	parser.add_argument('--n_trial', default=5, type=int)
	parser.add_argument('--n_iter', default=10, type=int)
	parser.add_argument('--n_batch', default=4, type=int)
	parser.add_argument('--n_init', default=10, type=int)
	
	parser.add_argument('--task_type', default='multi', type=str)
	parser.add_argument('--algo', default='qnehvi', type=str)
	parser.add_argument('--label', default='', type=str)
	
	args = parser.parse_args()
	main(args)


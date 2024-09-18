import torch
import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def LHS(problems, n_samples):

	space = Space(
		[(problems[0].bounds[0,i].item(), problems[0].bounds[1,i].item()) 
		 for i in range(problems[0].bounds.shape[1])]
		+ [(0, len(problems)-1)])
	
	lhs = Lhs(criterion="maximin", iterations=10000)
	lhs_x = lhs.generate(space.dimensions, n_samples)
	lhs_x = np.array(lhs_x)
	
	x = torch.tensor(lhs_x[:,:-1], **tkwargs)
	task = torch.tensor(lhs_x[:,-1], **tkwargs).unsqueeze(1)
	y = problems[0].evaluate(x[(task==0).all(dim=1)])
	
	for i in range(1, len(problems)):
		y = torch.vstack([y, problems[i].evaluate(x[(task==i).all(dim=1)])])
	
	return x, task, y
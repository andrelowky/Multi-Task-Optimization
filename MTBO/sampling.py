import torch
import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs

from botorch.utils.sampling import draw_sobol_samples


tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def sobol_repeated(problems, n_samples, random_state):

	x = draw_sobol_samples(bounds=problems[0].bounds, n=n_samples, q=1).squeeze(1).tile((len(problems),1))
	task = torch.tensor([np.repeat(i, n_samples) for i in range(len(problems))], **tkwargs).reshape(-1,1)
	
	y = problems[0].evaluate(x[(task==0).all(dim=1)])
	    
	for i in range(1, len(problems)):
	    y = torch.vstack([y, problems[i].evaluate(x[(task==i).all(dim=1)])])

	return x, task, y

def sobol(problems, n_samples, random_state):

	x = draw_sobol_samples(bounds=problems[0].bounds, n=n_samples*len(problems), q=1).squeeze(1)
	task = torch.tensor(np.arange(len(problems)).repeat(n_samples), **tkwargs).unsqueeze(1)
	
	y = problems[0].evaluate(x[(task==0).all(dim=1)])
		
	for i in range(1, len(problems)):
	    y = torch.vstack([y, problems[i].evaluate(x[(task==i).all(dim=1)])])

	return x, task, y
	
def LHS(problems, n_samples, random_state):

	space = Space(
		[(problems[0].bounds[0,i].item(), problems[0].bounds[1,i].item()) 
		 for i in range(problems[0].bounds.shape[1])])
	
	lhs = Lhs(criterion="maximin", iterations=10000)
	lhs_x = lhs.generate(space.dimensions, n_samples, random_state)
	lhs_x = np.array(lhs_x)
	lhs_x = np.tile(lhs_x, (len(problems),1))
	x = torch.tensor(lhs_x, **tkwargs)
	
	task = np.repeat(np.arange(len(problems)), n_samples, axis=0)
	task = torch.tensor(task, **tkwargs).unsqueeze(1)
		
	y = problems[0].evaluate(x[(task==0).all(dim=1)])
	
	for i in range(1, len(problems)):
		y = torch.vstack([y, problems[i].evaluate(x[(task==i).all(dim=1)])])
	
	return x, task, y
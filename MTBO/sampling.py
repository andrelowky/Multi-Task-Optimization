import torch
import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.stats import qmc

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}
	
def LHS(problems, n_samples, random_state):

	space = Space(
	    [(problems[0].bounds[0,i].item(), problems[0].bounds[1,i].item()) 
	     for i in range(problems[0].bounds.shape[1])]+[[0, int(len(problems))]])
	
	lhs = Lhs(criterion="maximin", iterations=10000)
	lhs_x1 = lhs.generate(space.dimensions, int(n_samples/2), random_state)
	lhs_x1 = np.array(lhs_x1)
	
	space = Space(
	    [(problems[0].bounds[0,i].item(), problems[0].bounds[1,i].item()) 
	     for i in range(problems[0].bounds.shape[1])])
	
	lhs = Lhs(criterion="maximin", iterations=10000)
	lhs_x2 = lhs.generate(space.dimensions, int(n_samples/(2*len(problems))), random_state)
	lhs_x2 = np.array(lhs_x2)
	lhs_x2 = np.tile(lhs_x2, (len(problems), 1))
	repeat_task = np.repeat(np.arange(len(problems)), int(n_samples/(2*len(problems)))).reshape(-1,1)
	lhs_x2 = np.hstack([lhs_x2, repeat_task])
	
	lhs_x = np.vstack([lhs_x1, lhs_x2])
	x = torch.tensor(lhs_x[:,:-1], **tkwargs)
	task = torch.tensor(lhs_x[:,-1], **tkwargs).unsqueeze(1)
	
	y = problems[0].evaluate(x[(task==0).all(dim=1)])
	    
	for i in range(1, len(problems)):
	    y = torch.vstack([y, problems[i].evaluate(x[(task==i).all(dim=1)])])
	
	return x, task, y

def anchored_sampling(problems, n_samples, random_state):
    np.random.seed(random_state)
    
    base_samples = int(n_samples/(2*len(problems)))
    remain_samples = int(n_samples/2)
    power_2 = int(np.ceil(np.log(base_samples+remain_samples)/np.log(2)))
    
    sampler = qmc.Sobol(d=problems[0].n_var, scramble=True)
    samples = sampler.random_base2(m=power_2)
    sample1 = samples[:base_samples]
    x1 = np.tile(sample1, (len(problems), 1))
    x2 = samples[base_samples:base_samples+int(n_samples/2)]
    tasks = np.repeat(np.arange(len(problems)), base_samples).reshape(-1,1)
    
    x = torch.tensor(np.vstack([x1, x2]), **tkwargs)
    task = torch.tensor(np.vstack([tasks, tasks]), **tkwargs)
    
    y = problems[0].evaluate(x[(task==0).all(dim=1)])
        
    for i in range(1, len(problems)):
        y = torch.vstack([y, problems[i].evaluate(x[(task==i).all(dim=1)])])

    return x, task, y
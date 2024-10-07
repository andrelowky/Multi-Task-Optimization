import torch
import math
import numpy as np

from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

MC_SAMPLES = 256

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def mt_qnparego(model, x, task, batch_size, n_obj):

    with torch.no_grad():
        pred = model.posterior(torch.hstack([x, task])).mean
    
    acq_func_list = []
    for i in range(batch_size):
        weights = sample_simplex(n_obj, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
        mtgp_acqf = qLogNoisyExpectedImprovement(
			model=model,
			X_baseline=torch.hstack([x, task]),
			objective=objective,
			prune_baseline=True, 
		)
        acq_func_list.append(mtgp_acqf)

    return acq_func_list

def mt_qucb(model, x, task, batch_size, n_obj):

    with torch.no_grad():
        pred = model.posterior(torch.hstack([x, task])).mean

    acq_func_list = []
    for i in range(batch_size):
        weights = sample_simplex(n_obj, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
        stgp_acqf = qUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                    objective=objective,
					sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),)
        acq_func_list.append(stgp_acqf)

    return acq_func_list
    
def mt_qnehvi(model, ref_pt, x, task):

    acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_pt.tolist(),
            X_baseline=torch.hstack([x, task]),
            prune_baseline=True,  
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        )
    
    return acq_func

def st_qnparego(model, x, batch_size, n_obj):

    with torch.no_grad():
        pred = model.posterior(x).mean
    
    acq_func_list = []
    for i in range(batch_size):
        weights = sample_simplex(n_obj, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
        stgp_acqf = qLogNoisyExpectedImprovement(
			model=model,
			X_baseline=x,
			objective=objective,
			prune_baseline=True)
        acq_func_list.append(stgp_acqf)
		
    return acq_func_list

def st_qnehvi(model, ref_pt, x):
    
    acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_pt.tolist(),
            X_baseline=x,
            prune_baseline=True,  
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        )

    return acq_func

def st_qucb(model, x, batch_size, n_obj):

    with torch.no_grad():
        pred = model.posterior(x).mean

    acq_func_list = []
    for i in range(batch_size):
        weights = sample_simplex(n_obj, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
        stgp_acqf = qUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                    objective=objective,
					sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),)
        acq_func_list.append(stgp_acqf)

    return acq_func_list

def ftgp_qucb(model, x, task, batch_size, n_obj):
	
	n_task = int(task.max())+1
	
	with torch.no_grad():
		pred = model.posterior(torch.hstack([x, task])).mean

	acq_func_list = []
	for i in range(batch_size*n_task):
		weights = sample_simplex(n_obj, **tkwargs).squeeze()
		objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred)).to(device)
		mtgp_acqf = qUpperConfidenceBound(model=model, objective=objective,
										 beta=0.1, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])))
		acq_func_list.append(mtgp_acqf)
	
	return acq_func_list

def get_total_weights(task, batch_size, n_obj):

    n_task = int(task.max())+1
    # first create the weights for each task
    weights = []
    
    for i in range(n_task):
        indices = [j * n_task + i for j in range(n_obj)]
        
        for _ in range(batch_size):
            sample_tensor = torch.zeros(n_obj * n_task, **tkwargs)
            simplex_point = torch.tensor(np.random.dirichlet(np.ones(n_obj)), **tkwargs)
            sample_tensor[indices] = simplex_point
            weights.append(sample_tensor)

    return weights
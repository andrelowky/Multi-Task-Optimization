import torch
import math
import numpy as np

from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import LinearMCObjective, GenericMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

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
                    objective=objective)
        acq_func_list.append(mtgp_acqf)

    return acq_func_list
    
def mt_qnehvi(model, ref_pt, x, task):

    acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_pt.tolist(),
            X_baseline=torch.hstack([x, task]),
            prune_baseline=True,  
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
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
                    objective=objective)
        acq_func_list.append(stgp_acqf)
		
    return acq_func_list

def st_qnehvi(model, ref_pt, x):
    
    acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_pt.tolist(),
            X_baseline=x,
            prune_baseline=True,  
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )

    return acq_func

def st_qucb(model, x, batch_size, n_obj):

    with torch.no_grad():
        pred = model.posterior(x).mean
    
    acq_func_list = []
    for i in range(batch_size):
        weights = sample_simplex(n_obj, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))

    acq_func_list = []
    for i in range(batch_size):
        weights = sample_simplex(n_obj, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
        stgp_acqf = qUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                    objective=objective,
					sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),)
        acq_func_list.append(stgp_acqf)

    return acq_func_list
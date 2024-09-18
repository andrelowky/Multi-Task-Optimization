

import torch

from botorch.models.multitask import MultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

def initialize_model_mt(x, task, y):

    models = []
    for i in range(y.shape[-1]):
        models.append(MultiTaskGP(torch.hstack([x, task.type_as(x)]),
                                  y[..., i : i + 1],
                                  task_feature=-1, 
                                  outcome_transform=Standardize(m=1)))

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return model, mll

def initialize_model_st(x, y):

    models = []
    for i in range(y.shape[-1]):
        models.append(SingleTaskGP(x, y[..., i : i + 1],
                                   outcome_transform=Standardize(m=1)))

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return model, mll
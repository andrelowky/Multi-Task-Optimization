import gpytorch
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from torchmin import Minimizer

from gpytorch.priors import Prior
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.distributions import MultivariateNormal

from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel

from gpytorch.constraints.constraints import GreaterThan, Interval, Positive
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from linear_operator.operators import DiagLinearOperator, InterpolatedLinearOperator, PsdSumLinearOperator, RootLinearOperator

from botorch.models.transforms.outcome import Standardize
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.gpytorch import GPyTorchModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransferKernel(Kernel):
    r"""
    A transfer kernel for TGP

    .. math::

        \begin{equation}
            k(i, j) = [[1, lambda],
                       [lambda, 1]]
        \end{equation}

    Attributes:
        covar_factor:
            The :math:`B` matrix.
    
    Note: This transfer kernel only supports two tasks.
    """    

    def __init__(
        self,
        lam = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if lam is not None:
            self.similarity = torch.tensor(lam)
        else:
            var_constraint = Interval(upper_bound=1,lower_bound=-1)
            self.register_parameter(
                name="raw_similarity", parameter=torch.nn.Parameter(1.0986*torch.ones(*self.batch_shape, 1))
                )
            self.register_constraint("raw_similarity", var_constraint)

    @property
    def similarity(self):
        return self.raw_similarity_constraint.transform(self.raw_similarity)

    @similarity.setter
    def similarity(self, value):
        self._set_similarity(value)

    def _set_similarity(self, value):
        self.initialize(raw_similarity=self.raw_similarity_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        var1 = torch.tensor([[0,1],[1,0]], device=device)*self.similarity
        var2 = torch.tensor([[1,0],[0,1]], device=device)
        return var1+var2

    @property
    def covar_matrix(self):
        var1 = torch.tensor([[0,1],[1,0]], device=device)*self.similarity
        var2 = torch.tensor([[1,0],[0,1]], device=device)
        res = InterpolatedLinearOperator(var1+var2)
        return res

    def forward(self, i1, i2, **params):

        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = torch.broadcast_shapes(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLinearOperator(
            base_linear_op=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood='gaussian', kernel='RBF'):  
        if likelihood == 'gaussian':
            likelihood_model = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-4,upper_bound=5))
        elif likelihood == 'gaussian_with_gamma_prior':
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood_model = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    1e-4,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_model)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'ARDMatern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(1))
        elif kernel == 'ARDMatern52_with_gamma_prior':
            self.covar_module = MaternKernel(
                                    nu=2.5,
                                    ard_num_dims=train_x.size(1),
                                    lengthscale_prior=GammaPrior(3.0, 6.0)
                                )
        elif kernel == 'ARDRBF':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1))
        elif kernel == 'ARDRBF_with_gamma_prior':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1), lengthscale_prior=GammaPrior(3.0, 6.0))
        elif kernel == 'Matern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel == 'RBF':
            self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=GreaterThan(1e-4))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TransferGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBF'):
        if likelihood == 'gaussian':
            likelihood_model = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-4,upper_bound=5))
        elif likelihood == 'gaussian_with_gamma_prior':
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood_model = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    1e-4,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )

        super(TransferGPModel, self).__init__(train_x, train_y, likelihood_model)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'ARDMatern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x[0].size(1))
        elif kernel == 'ARDMatern52_with_gamma_prior':
            self.covar_module = MaternKernel(
                                    nu=2.5,
                                    ard_num_dims=train_x[0].size(1),
                                    lengthscale_prior=GammaPrior(3.0, 6.0)
                                )                                
        elif kernel == 'ARDRBF':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x[0].size(1))
        elif kernel == 'ARDRBF_with_gamma_prior':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x[0].size(1),lengthscale_prior=GammaPrior(3.0, 6.0))
        elif kernel == 'Matern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel == 'RBF':
            self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=GreaterThan(1e-4))

        self.task_covar_module = TransferKernel()

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class EnsembleFTGP(GPyTorchModel):
	def __init__(self, train_x, train_task, train_y, likelihood='gaussian', kernel='RBF'):
		super().__init__()
		self.kernel = kernel
		self.likelihood_type = likelihood
			
		self.outcome_transform = Standardize(m=1)
		
		self.train_x = train_x
		self.train_task = train_task.squeeze(1)
		self.train_y, _ = self.outcome_transform(train_y)
		
		self.n_tasks = len(torch.unique(train_task))
		self.n_var = train_x.shape[1]
		self._num_outputs = 1
		
		self.n_epoch = 200

	@property
	def num_outputs(self) -> int:
		return self._num_outputs
	
	@num_outputs.setter
	def num_outputs(self, val: int) -> None:
		self._num_outputs = val
		
	def build_single_task_gps(self):
		self.models_st = []
		for i in range(self.n_tasks):

			mask = (self.train_task == i)
			x = self.train_x[mask]
			y = self.train_y[mask].squeeze(1)
			model = ExactGPModel(x, y, self.likelihood_type, kernel=self.kernel).to(device)
			model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

			model.train()
			model.likelihood.train()
			
			mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
			
			optimizer = Minimizer(model.parameters(),method='l-bfgs',
								  tol=1e-6, max_iter=200, disp=0)
			def closure():
				optimizer.zero_grad()
				output = model(x)
				loss = -mll(output,y)
				# loss.backward()
				return loss.sum()
			optimizer.step(closure = closure)

			model.eval()
			self.models_st.append(model)
				
	def build_transfer_gps(self):
		self.models_mt = []
		self.pairs = []
		
		numbers = np.arange(self.n_tasks)
		for target in numbers:
			for source in numbers:
				if target != source:
					
					mask_i = (self.train_task == target)
					mask_j = (self.train_task == source)
					
					x = torch.cat([self.train_x[mask_i], self.train_x[mask_j]], 0).to(device)
					y = torch.cat([self.train_y[mask_i], self.train_y[mask_j]], 0).squeeze(1).to(device)
					task = torch.cat([torch.zeros(mask_i.sum(), dtype=torch.long),
									  torch.ones(mask_j.sum(), dtype=torch.long)], 0).to(device)
					model = TransferGPModel((x, task), y, 
											likelihood=self.likelihood_type, kernel=self.kernel).to(device)
					model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

					model.train()
					model.likelihood.train()

					# transfer model inherits the parameters of its single-task base
					model.mean_module.constant.data = self.models_st[target].mean_module.constant.data
					model.covar_module.lengthscale = self.models_st[target].covar_module.lengthscale
					model.likelihood.noise_covar.raw_noise = self.models_st[target].likelihood.noise_covar.raw_noise
					model.mean_module.constant.requires_grad = False
					model.covar_module.raw_lengthscale.requires_grad = False
					model.likelihood.noise_covar.raw_noise.requires_grad = False

					mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
					optimizer = Minimizer(model.parameters(),method='l-bfgs',
										  tol=1e-6, max_iter=200, disp=0)
					def closure():
						optimizer.zero_grad()
						output = model(x, task)
						loss = -mll(output,y)
						# loss.backward()
						return loss.sum()
					optimizer.step(closure = closure)

					model.eval()
					self.models_mt.append(model)
					self.pairs.append((target, source))
		
	def fit(self):
		self.build_single_task_gps()
		self.build_transfer_gps()

		self.likelihood = gpytorch.likelihoods.LikelihoodList(
			*[model.likelihood for model in self.models_st + self.models_mt]
		)
	
	def find_indexes_of_first(self, num, tuple_list):
		indexes = [index for index, (first, _) in enumerate(tuple_list) if first == num]
		return indexes
		
	def predict(self, X):

		transform_y = False

		if X.dim() == 3:
			n_samples, n_batches, _ = X.shape
			X = X.reshape(-1, self.n_var+1)
			transform_y = True
			
		x = X[:, :-1] # All columns except the last one
		task = X[:, -1:]
	
		means, variances = [], []
	
		for target in range(self.n_tasks):
			mask = (task == target).all(dim=1)
			x_mask = x[mask]
			if x_mask.numel() == 0:
				continue  # Skip if no points for this task
			
			indexes = self.find_indexes_of_first(target, self.pairs)
			predictions_s = []
			test_i_task = torch.zeros(x_mask.size(0), dtype=torch.long).to(device)
			for idx in indexes:
				predictions_s.append(self.models_mt[idx](x_mask, test_i_task))          
			prediction_t = self.models_st[target](x_mask)
			
			M = len(predictions_s)
			# Prediction of the factorized TGP
			predicted_variance_inv_temp = torch.stack([1 / predictions_s[j].variance for j in range(M)]).sum(axis=0)
			predicted_variance_inv = predicted_variance_inv_temp + (1 - M) * 1.0 / prediction_t.variance
			predicted_variance_FTGP = 1 / predicted_variance_inv
			predicted_mean_FTGP = 0
			for j in range(M):
				predicted_mean_FTGP += (predictions_s[j].mean / predictions_s[j].variance)
			predicted_mean_FTGP += (1 - M) * (prediction_t.mean / prediction_t.variance)
			predicted_mean_FTGP = predicted_mean_FTGP * predicted_variance_FTGP
			
			# If the standard dev provided by factorized TGP is not positive, provided the predictions by using generalized product of experts (GPOE)
			predicted_variance_GPOE = M / predicted_variance_inv_temp
			predicted_mean_GPOE = 0
			for j in range(M):
				predicted_mean_GPOE += (predictions_s[j].mean / predictions_s[j].variance)
			predicted_mean_GPOE = predicted_mean_GPOE * predicted_variance_GPOE / M
			
			# If not positive, use GPOE
			mean = (predicted_variance_inv > 0) * predicted_mean_FTGP + (predicted_variance_inv <= 0) * predicted_mean_GPOE
			variance = (predicted_variance_inv > 0) * predicted_variance_FTGP + (predicted_variance_inv <= 0) * predicted_variance_GPOE
			
			mean, variance = self.outcome_transform.untransform(mean, variance)
			means.append(mean)
			variances.append(variance)

		# Combine predictions for all tasks
		if transform_y:
			means = torch.cat(means, dim=1).reshape(n_samples, n_batches, 1).to(device)
			variances = torch.cat(variances, dim=1).reshape(n_samples, n_batches, 1).to(device)
		else:
			means = torch.cat(means, dim=1).reshape(-1,1).to(device)
			variances = torch.cat(variances, dim=1).reshape(-1,1).to(device)			
		
		return means, variances

	
	def posterior(self, X: torch.Tensor, output_indices: Optional[torch.Tensor] = None, 
				  observation_noise: bool = False, **kwargs: Any) -> GPyTorchPosterior:

		#X = X.reshape(-1, self.n_var+1)
		
		mean, variance = self.predict(X)
	
		mvn = MultivariateNormal(mean.squeeze(-1), torch.diag_embed(variance.squeeze(-1)))
		posterior = GPyTorchPosterior(mvn)
		return posterior
	
	@property
	def num_outputs(self) -> int:
		return self._num_outputs
	
	@num_outputs.setter
	def num_outputs(self, val: int) -> None:
		self._num_outputs = val
	
	def forward(self, X: torch.Tensor) -> MultivariateNormal:
		mean, variance = self.predict(X[:, :-1], X[:, -1:])

		return MultivariateNormal(mean.squeeze(-1), torch.diag_embed(variance.squeeze(-1)))
		
	@property
	def batch_shape(self) -> torch.Size:
		return torch.Size([])  # Assuming no batching for now
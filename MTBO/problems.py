import torch
import numpy as np
import math

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

class DTLZ1():
    def __init__(self, n_var = 10, delta1 = 1, delta2 = 0, delta3 = 1, negate=True):
        self.n_var = n_var
        self.n_obj = 3
        self.negate = negate

        bounds = torch.zeros((2, n_var), **tkwargs)
        bounds[1] = 1
        self.bounds = bounds

        if self.negate:
           self.ref_pt = torch.tensor([-400, -400, -400], **tkwargs) 
        else:
            self.ref_pt = torch.tensor([400, 400, 400], **tkwargs)
                
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
    
    def evaluate(self, x):

        M = 3
        g = 100*self.delta3*(8 + torch.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2 - torch.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)), axis=1 )) + (1-self.delta3)*(-20*torch.exp(-0.2*torch.sqrt(torch.mean( ((x[:,M-1:] -0.5 - self.delta2)*50)**2, axis=1 ) )) - torch.exp(torch.mean(torch.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)*50) , axis=1 ) ) + 20 + np.e)

        f1 = 0.5*self.delta1*self.delta3*x[:,0]*x[:,1]*(1+g) + (1-self.delta3)*(1+g)*torch.cos(x[:,0]*np.pi/2)*torch.cos(x[:,1]*np.pi/2)
        f2 = 0.5*self.delta1*self.delta3*x[:,0]*(1-x[:,1])*(1+g) + (1-self.delta3)*(1+g)*torch.cos(x[:,0]*np.pi/2)*torch.sin(x[:,1]*np.pi/2)
        f3 = 0.5*self.delta1*self.delta3*(1-x[:,0])*(1+g) + (1-self.delta3)*(1+g)*torch.sin(x[:,0]*np.pi/2)

        if self.negate:
            return -1* torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)])
        else:
            return torch.hstack([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)])
import torch

class KFACOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, step_size, damping):
        super().__init__(parameters, dict(step_size=step_size, damping=damping))
    
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                print(p.grad)
                print(p._backward_hooks)
                
        return loss
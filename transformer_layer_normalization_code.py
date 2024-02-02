import torch
from torch import nn

inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
B, S, E = inputs.size()
inputs = inputs.reshape(S, B, E)

parameter_shape = inputs.size()[-2:]
print(parameter_shape)
gamma = nn.Parameter(torch.ones(parameter_shape))
beta = nn.Parameter(torch.zeros(parameter_shape))   

print(gamma.size(), beta.size())

dims = [-(i + 1) for i in range(len(parameter_shape))]
print(dims)

mean = inputs.mean(dim=dims, keepdim=True)
print(mean.size())

var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
epsilon = 1e-5
std = (var + epsilon).sqrt()
print(std)

y = (inputs - mean) / std
print(y)

out = gamma * y + beta
print(out)



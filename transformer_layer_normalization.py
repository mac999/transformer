import torch
from torch import nn

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f'Mean size: {mean.size()}, Mean: {mean}')
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f'std size: {std.size()}, std: {std}')
        y = (inputs - mean) / std
        print(f'y size: {y.size()}, y: {y}')
        out = self.gamma * y + self.beta
        print(f'out size: {out.size()}, out: {out}')
        return out

batch_size = 3
sentense_length = 5 
embedding_dim = 8
inputs = torch.randn(sentense_length, batch_size, embedding_dim)
print(f'inputs size: {inputs.size()}, inputs: {inputs}')

layer_norm = LayerNormalization(parameters_shape=(embedding_dim,))
out = layer_norm.forward(inputs)
print(f'out size: {out.size()}, out: {out}')
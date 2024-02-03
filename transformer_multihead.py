import io, math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
	d_k = q.shape[-1]
	scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # QK^T / sqrt(d_k)
	if mask is not None:
		scaled += mask  # considering future predition token
	attention = F.softmax(scaled, dim=-1)
	values = torch.matmul(attention, v)   # QK^T
	return values, attention

class multihead_attention(nn.Module):
	def __init__(self, input_dim, d_model, num_heads):
		super().__init__()
		self.input_dim = input_dim
		self.d_model = d_model
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
		self.linear_layer = nn.Linear(d_model, d_model)

	def forward(self, x, mask=None):
		batch_size, sequence_length, input_dim = x.size()
		print(f'x.size(): {x.size()}')
		qkv = self.qkv_layer(x)
		qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
		qkv = qkv.permute(0, 2, 1, 3)
		q, k, v = qkv.chunk(3, dim=-1)
		values, attention = scaled_dot_product(q, k, v, mask)
		values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
		out = self.linear_layer(values)
		return out

input_dim = 1024
d_model = 512
num_heads = 8
batch_size = 30
sequence_length = 5
print(f'multihead attention.\ninput_dim={input_dim}, d_model={d_model}, num_heads={num_heads}, batch_size={batch_size}, sequence_length={sequence_length}')

x = torch.randn((batch_size, sequence_length, input_dim))
print(f'x={x}')
model = multihead_attention(input_dim, d_model, num_heads)
out = model.forward(x)
print(f'out={out}')



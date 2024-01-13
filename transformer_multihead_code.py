import io, math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# multi head of qkv
sequence_length = 4
batch_size = 1
input_dim = 512
d_model = 512
x = torch.randn((batch_size, sequence_length, input_dim))
print(x.size())

qkv_layer = nn.Linear(input_dim, 3 * d_model)
qkv = qkv_layer(x)
print(qkv.shape)

import matplotlib.pyplot as plt
y_val = torch.histc(qkv, bins=200, min=-3, max=3)
x_val = np.arange(-1, 1, 0.01) * 3
plt.bar(x_val, y_val, align='center', color=['forestgreen'])
plt.title('qkv distribution')
# plt.show()

num_heads = 8
head_dim = d_model // num_heads
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim) # 3 = qkv vector
print(qkv.shape)

qkv = qkv.permute(0, 2, 1, 3)
print(qkv.shape)

q, k, v = qkv.chunk(3, dim=-1)
print(q.shape, k.shape, v.shape)

# attention
d_k = q.size()[-1]
scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # transpose k's sequence and head matrix
print(scaled.shape)

mask = torch.full(scaled.size(), float('-inf'))
mask = torch.triu(mask, diagonal=1)
print(mask[0][1])   # mask for input to single head

scaled += mask
attention = F.softmax(scaled, dim=-1)   # dim=-1 is last dimension
print(attention[0][0])

values = torch.matmul(attention, v)
print(values.shape)




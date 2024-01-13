import numpy as np
import math

L, d_k, d_v = 4, 8, 8   # my name is tom
q = np.random.randn(L, d_k) # what I am looking for
k = np.random.randn(L, d_k) # what I can offer
v = np.random.randn(L, d_v) # what I actually offer

def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled = np.matmul(q, k.T) / math.sqrt(d_k) # QK^T / sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask  # considering future predition token
    attention = softmax(scaled)
    # softmax(QK^T / sqrt(d_k) + M)V
    out = np.matmul(attention, v)   # QK^T
    return out, attention

# masking. to ensure words can only attend to previous words. dont't look at future words
mask = np.tril(np.ones([L, L])) # lower triangular matrix
print(mask)

values, attention = scaled_dot_product_attention(q, k, v, mask=None)
print('Q\n', q)
print('K\n', k)
print('V\n', v)
print('New V\n', values)
print('Attention\n', attention)

# reference. https://www.youtube.com/watch?v=QCJQG4DuHT0&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4

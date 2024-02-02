import torch
import torch.nn as nn

max_sequence_length = 10
d_model = 6

event_i = torch.arange(0, d_model, 2).float()
print(event_i)

event_denominator = torch.pow(10000, event_i / d_model)
print(event_denominator)

odd_i = torch.arange(1, d_model, 2).float()
print(odd_i)

even_denominator = torch.pow(10000, (odd_i - 1) / d_model)
print(even_denominator)

denominator = event_denominator

position = torch.arange(max_sequence_length, dtype=torch.float).reshape(max_sequence_length, 1)

even_PE = torch.sin(position / denominator)
odd_PE = torch.cos(position / denominator)

print(even_PE)
print(odd_PE)

stacked = torch.stack([even_PE, odd_PE], dim=2)
PE = torch.flatten(stacked, start_dim=1, end_dim=2)
print(PE)
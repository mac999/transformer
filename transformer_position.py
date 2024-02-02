import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_sequence_length):
		super().__init__()
		self.max_sequence_length = max_sequence_length
		self.d_model = d_model

	def forward(self):
		event_i = torch.arange(0, self.d_model, 2).float()
		denominator = torch.pow(10000, event_i /self.d_model)
		position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
		even_PE = torch.sin(position / denominator)
		odd_PE = torch.cos(position / denominator)
		stacked = torch.stack([even_PE, odd_PE], dim=2)
		PE = torch.flatten(stacked, start_dim=1, end_dim=2)
		return PE

pe = PositionalEncoding(d_model=6, max_sequence_length=10)
PE = pe.forward()
print(PE)


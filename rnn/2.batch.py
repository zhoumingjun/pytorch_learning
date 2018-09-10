import torch

input_size = 10
hidden_size = 20
num_layers = 1

# model
model = torch.nn.LSTM(input_size, hidden_size, num_layers)

# data
input = torch.randn(4, 4, 10)

# option1: sequence
output, hidden = model(input)

# option2: step by step
input_0 = input[:, 0, :].view(4,1,10)
input_1 = input[:, 1, :].view(4,1,10)
input_2 = input[:, 2, :].view(4,1,10)
input_3 = input[:, 3, :].view(4,1,10)

output_0, hidden_0 = model(input_0)
output_1, hidden_1 = model(input_1)
output_2, hidden_2 = model(input_2)
output_3, hidden_3 = model(input_3)


print((output[-1][0]- output_0[-1][0]).sum())
print((output[-1][1]- output_1[-1][0]).sum())
print((output[-1][2]- output_2[-1][0]).sum())
print((output[-1][3]- output_3[-1][0]).sum())


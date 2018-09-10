import torch

input_size = 10
hidden_size = 20
num_layers = 1

# model
model = torch.nn.LSTM(input_size, hidden_size, num_layers)

# data
input = torch.ones(4, 1, 10)

# option1: sequence
output, hidden = model(input)

# option2: step by step
input_0 = input[0,:,:].view(1,1,10)
input_1 = input[1,:,:].view(1,1,10)
input_2 = input[2,:,:].view(1,1,10)
input_3 = input[3,:,:].view(1,1,10)

output_0, hidden_0 = model(input_0)
output_1, hidden_1 = model(input_1, hidden_0)
output_2, hidden_2 = model(input_2, hidden_1)
output_3, hidden_3 = model(input_3, hidden_2)


print(hidden)
print(output)
print(hidden_0, hidden_1, hidden_2,hidden_3)
print(output_0, output_1, output_2,output_3)


# compare option1 & option2
print ((output[0]==output_0).sum().item() == hidden_size)
print ((output[1]==output_1).sum().item() == hidden_size)
print ((output[2]==output_2).sum().item() == hidden_size)
print ((output[3]==output_3).sum().item() == hidden_size)

"""
True
True
True
True
"""
# relation between hidden & output
print ((output[0]==hidden_0[0][-1]).sum().item() == hidden_size)
print ((output[1]==hidden_1[0][-1]).sum().item() == hidden_size)
print ((output[2]==hidden_2[0][-1]).sum().item() == hidden_size)
print ((output[3]==hidden_3[0][-1]).sum().item() == hidden_size)
"""
True
True
True
True
"""
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

input_size = 2
hidden_size= 5
num_layers = 1
nClasses = 10
nSamples = 10

a = torch.ones(3, input_size)
b = torch.ones(5, input_size)
c = torch.ones(7, input_size)

# pad
pad = pad_sequence([c,b,a])
print("pad result", pad.size())

# pack
pack = pack_sequence([c,b,a])
print("pack result:", pack.data.size(), pack.batch_sizes)

# pack_padded
pack_padded = pack_padded_sequence(pad, [7,5,3])
print("pack_padded result:", pack_padded.data.size(), pack_padded.batch_sizes)

# pad_packed
pad_packed_data, pad_packed_lengths = pad_packed_sequence(pack)
print("pad_packed result:", pad_packed_data.size() ,pad_packed_lengths)

# pattern

"""
prepare data/model/indices
"""

# data
inputs = []
targets = []
for idx in range(nSamples):
    # set random len of input , and set the len as target
    # input: ones(len, input_size)
    # target: len
    len = np.random.randint(nSamples)+1
    sample = torch.ones(len, input_size)
    inputs.append(sample)
    targets.append(len)

# model
model = torch.nn.LSTM(input_size, hidden_size, num_layers)
demo = torch.ones(10,1,  input_size)
print("sample sequence result", model(demo)[0])

# indices
sample_length = [x.size(0) for x in inputs]
_, indices_sorted = torch.sort(torch.LongTensor(sample_length), descending=True)
_, indices_restore = torch.sort(indices_sorted)

print("sample length:", sample_length)

"""
option1:
pre-process inputs
sort (inputs)-> pack(inputs) -> rnn -> unpack -> unsort(outputs)

targets <-> outputs  
"""
print("option1")

 

# sort inputs
inputs_sorted = [inputs[x] for x in indices_sorted]

# pack inputs
pack = pack_sequence(inputs_sorted)

# rnn ...
outputs, hidden = model(pack)

# unpack
output_unpacked, unpack_outputs_length = pad_packed_sequence(outputs)
last_state = output_unpacked[unpack_outputs_length-1, [x for x in range(10)] ,:]

# unsort
unsorted_last_state = last_state[indices_restore,:]
print([(tup[0].size(0), tup[1], tup[2]) for tup in   zip(inputs, targets, unsorted_last_state)])

"""
option2 
pre-process (inputs, targets)
sort (inputs, targets)-> pack(inputs) -> rnn -> unpack

targets(sorted) <--> outputs  
"""

print("option2")
batch = list(zip(inputs, targets))

# sort inputs
batch_sorted = [batch[x] for x in indices_sorted]

# pack inputs
pack = pack_sequence([tup[0] for tup in batch_sorted])

# rnn ...
outputs, hidden = model(pack)

# unpack
output_unpacked, unpack_outputs_length = pad_packed_sequence(outputs)
last_state = output_unpacked[unpack_outputs_length-1, [x for x in range(10)] ,:]

print([(tup[0][0].size(0), tup[0][1], tup[1]) for tup in zip(batch_sorted, last_state)])



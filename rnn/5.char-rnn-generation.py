import glob
import os
from torch.utils.data import Dataset, DataLoader
from os import path
import unicodedata
import torch
import string

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

# dataset
class NamesDataset(Dataset):
    def __init__(self, data_dir, transforms=[]):
        self.data_dir = data_dir


        inputs = []


        filepath = path.join(data_dir, "English.txt")

        with open(filepath) as f:
            lines = f.readlines()
            inputs += [line.strip() for line in lines]



        self.inputs = inputs
        self.transforms = transforms

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        item = self.inputs[idx]


        return item



# transform
class UnicodeToAscii(object):
    def __init__(self, letters):
        self.letters = letters

    def __call__(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.letters
        )

# ascii name to tenser
def name2tensor(s):
    tensor = torch.zeros(len(s), n_letters)
    for li, letter in enumerate(s):
        tensor[li][all_letters.index(letter)] = 1
    return tensor



class NamesClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NamesClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = torch.nn.LSTM(input_size, hidden_size, 1)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, hidden = self.lstm(x)
        x = x.squeeze()
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


unicode2ascii = UnicodeToAscii(all_letters)

# define dataset and dataloader
namesDataset = NamesDataset('./data/names')
dataloader = DataLoader(namesDataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
transforms = [unicode2ascii]
# hyper parameters
input_size = n_letters
hidden_size = 50
output_size = input_size

# define model/optimizer/criterion
model = NamesClassifier(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()


for epoch in range(10):

    # train
    loss_sum = 0;
    nRound = 0
    for i_batch, batch in enumerate(dataloader):

        # zero

        inputs = batch

        for idx, input in enumerate(inputs):

            for transform in transforms:
                input = transform(input)

            label = input[1:]+all_letters[-1]

            optimizer.zero_grad()


            input = name2tensor(input).view(-1, 1, input_size)
            label = [all_letters.index(x) for x in label]

            output = model(input)
            loss = criterion(output, torch.LongTensor(label))

            loss.backward()
            optimizer.step()

            loss_sum += loss
            nRound +=1


    print("epoch {} i_batch {} loss {}".format(epoch, i_batch, loss_sum / nRound))
#
#     # validate
#     with torch.no_grad():
#         acc = 0
#         for i_batch, batch in enumerate(dataloader):
#             inputs, labels = batch
#             # pre-process
#             inputs = [name2tensor(name) for name in inputs]
#
#             inputs_length = [x.size(0) for x in inputs]
#             _, indices_sorted = torch.sort(torch.LongTensor(inputs_length), descending=True)
#             _, indices_restore = torch.sort(indices_sorted)
#
#             # sort
#             inputs_sorted = [inputs[x] for x in indices_sorted]
#             labels_sorted = labels[indices_sorted]
#
#             # pack inputs
#             pack = pack_sequence(inputs_sorted)
#
#             # rnn
#             outputs = model(pack)
#
#             top_v, topi = torch.topk(outputs, 1)
#             acc += (topi.view(1, -1) == labels_sorted).sum().item()
#
#     print("epoch {} acc:{}/{} ".format(epoch, acc, len(namesDataset)))
#
# # do some preidct
# for i_batch, batch in enumerate(dataloader):
#     input, label = batch
#     for idx, input in enumerate(input):
#         lang, lang_id = predict(input)
#         print("input {}, label {}, predict {}, result: {}".format(
#             input, all_langs[label[idx].item()],
#             lang,
#             lang_id == label[idx].item()))
#
#     break

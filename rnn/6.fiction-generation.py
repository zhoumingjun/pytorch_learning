from torch.utils.data import Dataset, DataLoader
from os import path
import unicodedata
import torch
import string

import torch.nn.functional as F
import re

import numpy as np


class NamesDataset(Dataset):
    def __init__(self, data_path):

        inputs = []
        all_letters = set([])
        with open(data_path) as f:
            while True:
                seq = f.read(10)
                if not seq or len(seq) != 10:
                    print("End of file")
                    break

                all_letters = all_letters | set(seq)
                inputs.append(seq)

        self.inputs = inputs
        self.all_letters = [x for x in all_letters]
        self.n_letters = len(self.all_letters)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def getAllLetters(self):
        return self.all_letters

    def getNLetters(self):
        return self.n_letters

    def name2tensor(self, s):
        tensor = torch.zeros(len(s), self.n_letters)
        for li, letter in enumerate(s):
            tensor[li][self.all_letters.index(letter)] = 1
        return tensor

    def name2Class(self, s):
        return [self.all_letters.index(x) for x in s]

    def names2tensor(self, s):
        batch = len(s)

        seq = len(s[0])

        tensor = torch.zeros(seq, batch, self.n_letters)
        for x, name in enumerate(s):
            for y, ch in enumerate(name):
                tensor[y][x][self.all_letters.index(ch)] = 1
        return tensor

    def names2Class(self, s):

        batch = len(s)
        seq = len(s[0])
        tensor = torch.zeros(seq, batch, dtype=torch.long)

        for x, name in enumerate(s):
            for y, ch in enumerate(name):
                tensor[y][x] = self.all_letters.index(ch)
        return tensor


dataset = NamesDataset("./data/cun.txt")
print("fiction length:", len(dataset) * 1000)
print("letters stat:", dataset.getNLetters(), dataset.getAllLetters())
# print("name2class", dataset.name2Class("iam"))
# print("name2tensor", dataset.name2tensor("Iam"))

dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4, drop_last=False)


class FictionGenerator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FictionGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = torch.nn.LSTM(input_size, hidden_size, 1)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hx=hidden)
        x = x.view(-1, self.hidden_size)
        x = self.linear(x)
        return F.log_softmax(x, dim=1), hidden


input_size = dataset.n_letters
hidden_size = 500
output_size = dataset.n_letters

# define model/optimizer/criterion
model = FictionGenerator(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)


def sample(start=1, limit=1000):
    with torch.no_grad():  # no need to track history in sampling
        letter = dataset.all_letters[start]
        fiction = letter

        input = dataset.name2tensor(letter).view(1, 1, input_size).to(device)
        hidden = (torch.zeros(1, 1, hidden_size).to(device), torch.zeros(1, 1, hidden_size).to(device))

        for i in range(limit):
            output, hidden = model(input, hidden)
            p = torch.exp(output).squeeze().cpu().numpy()

            # random choice according to prob
            ix = np.random.choice(range(dataset.n_letters), p=p)

            # max prob
            # topv, topi = output.topk(1)
            # ix = topi[0][0]
            #
            # print(ix, topi)

            letter = dataset.all_letters[ix]
            fiction += letter

            input = dataset.name2tensor(letter).view(-1, 1, input_size).to(device)

        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(fiction)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return fiction


for epoch in range(1000):

    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # prepare data
        input = [x[:-1] for x in batch]
        label = [x[1:] for x in batch]
        inputs = dataset.names2tensor(input).to(device)
        labels = dataset.names2Class(label).view(-1).to(device)

        # train
        output, _ = model(inputs, None)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    print("epoch {} i_batch {} loss {}".format(epoch, i_batch, loss))
    sample()

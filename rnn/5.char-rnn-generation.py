from torch.utils.data import Dataset, DataLoader
from os import path
import unicodedata
import torch
import string

import torch.nn.functional as F

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

# ascii name to tenser
def name2tensor(s):
    tensor = torch.zeros(len(s), n_letters)
    for li, letter in enumerate(s):
        tensor[li][all_letters.index(letter)] = 1
    return tensor

# dataset
class NamesDataset(Dataset):
    def __init__(self, data_path, transforms=[]):
        inputs = []
        filepath = path.join(data_path)

        with open(filepath) as f:
            lines = f.readlines()
            inputs += [line.strip() for line in lines]

        self.inputs = inputs
        self.transforms = transforms

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        for transform in self.transforms:
            input = transform(input)
        return input


class UnicodeToAscii(object):
    def __init__(self, letters):
        self.letters = letters

    def __call__(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.letters
        )


namesDataset = NamesDataset('./data/names/English.txt', transforms=[UnicodeToAscii(all_letters)])


class NamesClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NamesClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = torch.nn.LSTM(input_size, hidden_size, 1)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hx=hidden)
        x = x.view(-1, hidden_size)
        x = self.linear(x)
        return F.log_softmax(x, dim=1), hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


dataloader = DataLoader(namesDataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

input_size = n_letters
hidden_size = 50
output_size = input_size

# define model/optimizer/criterion
model = NamesClassifier(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

max_length = 20
def sample(start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        input = name2tensor(start_letter).view(-1, 1, input_size)
        output_name = start_letter
        hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1,1,hidden_size))

        for i in range(max_length):
            output, hidden = model(input,hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter

            input = name2tensor(letter).view(-1, 1, input_size)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(start_letter))


for epoch in range(20):

    loss_sum = 0;
    nRound = 0
    for i_batch, batch in enumerate(dataloader):

        inputs = batch

        for idx, input in enumerate(inputs):
            label = input[1:] + all_letters[-1]

            optimizer.zero_grad()

            input = name2tensor(input).view(-1, 1, input_size)
            label = [all_letters.index(x) for x in label]

            output, _ = model(input,  None)
            loss = criterion(output, torch.LongTensor(label))

            loss.backward()
            optimizer.step()

            loss_sum += loss
            nRound += 1

    print("epoch {} i_batch {} loss {}".format(epoch, i_batch, loss_sum / nRound))
    samples()








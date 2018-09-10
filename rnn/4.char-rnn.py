import glob
import os
from torch.utils.data import Dataset, DataLoader
from os import path
import unicodedata
import torch
import string

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# dataset
class NamesDataset(Dataset):
    def __init__(self, data_dir, transforms=[]):
        self.data_dir = data_dir

        all_langs = []
        inputs = []
        labels = []

        for filepath in glob.glob(path.join(data_dir, "*")):
            lang = os.path.splitext(os.path.basename(filepath))[0]
            if not lang in all_langs:
                all_langs.append(lang)

            label = all_langs.index(lang)

            with open(filepath) as f:
                lines = f.readlines()
                inputs += [line.strip() for line in lines]
                labels += [label] * len(lines)

        self.all_langs = all_langs
        self.inputs = inputs
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        item = self.inputs[idx]
        for transform in self.transforms:
            item = transform(item)

        return item, self.labels[idx]

    def getLangs(self):
        return self.all_langs


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


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_langs[category_i], category_i


def predict(s):
    with torch.no_grad():
        ascii = unicode2ascii(s)
        input = name2tensor(ascii)
        pack = pack_sequence([input])
        output = model(pack)
        return categoryFromOutput(output)

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
        output_unpacked, unpack_outputs_length = pad_packed_sequence(x)

        seqs = unpack_outputs_length - 1
        batch = [x for x in range(len(unpack_outputs_length))]
        last_state = output_unpacked[seqs, batch, :].view(-1, self.hidden_size)

        x = self.linear(last_state)
        return F.log_softmax(x, dim=1)


unicode2ascii = UnicodeToAscii(all_letters)

# define dataset and dataloader
namesDataset = NamesDataset('./data/names', transforms=[unicode2ascii])
dataloader = DataLoader(namesDataset, batch_size=100, shuffle=True, num_workers=4, drop_last=True)

# hyper parameters
input_size = n_letters
hidden_size = 50
all_langs = namesDataset.getLangs()
output_size = len(all_langs)

# define model/optimizer/criterion
model = NamesClassifier(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

print("all_langs", all_langs)

for epoch in range(10):

    # train
    loss_sum = 0;
    nRound = 0
    for i_batch, batch in enumerate(dataloader):

        # zero
        optimizer.zero_grad()

        inputs, labels = batch
        # pre-process
        inputs = [name2tensor(name) for name in inputs]

        inputs_length = [x.size(0) for x in inputs]
        _, indices_sorted = torch.sort(torch.LongTensor(inputs_length), descending=True)
        _, indices_restore = torch.sort(indices_sorted)

        # sort
        inputs_sorted = [inputs[x] for x in indices_sorted]
        labels_sorted = labels[indices_sorted]

        # pack inputs
        pack = pack_sequence(inputs_sorted)

        # rnn
        outputs = model(pack)

        # loss/bp/step
        loss = criterion(outputs, labels_sorted)

        loss.backward()
        optimizer.step()

        loss_sum += loss
        nRound += 1
        if i_batch % 50 == 0:
            print("epoch {} i_batch {} loss {}".format(epoch, i_batch, loss_sum / nRound))

    # validate
    with torch.no_grad():
        acc = 0
        for i_batch, batch in enumerate(dataloader):
            inputs, labels = batch
            # pre-process
            inputs = [name2tensor(name) for name in inputs]

            inputs_length = [x.size(0) for x in inputs]
            _, indices_sorted = torch.sort(torch.LongTensor(inputs_length), descending=True)
            _, indices_restore = torch.sort(indices_sorted)

            # sort
            inputs_sorted = [inputs[x] for x in indices_sorted]
            labels_sorted = labels[indices_sorted]

            # pack inputs
            pack = pack_sequence(inputs_sorted)

            # rnn
            outputs = model(pack)

            top_v, topi = torch.topk(outputs, 1)
            acc += (topi.view(1, -1) == labels_sorted).sum().item()

    print("epoch {} acc:{}/{} ".format(epoch, acc, len(namesDataset)))

# do some preidct
for i_batch, batch in enumerate(dataloader):
    input, label = batch
    for idx, input in enumerate(input):
        lang, lang_id = predict(input)
        print("input {}, label {}, predict {}, result: {}".format(
            input, all_langs[label[idx].item()],
            lang,
            lang_id == label[idx].item()))

    break

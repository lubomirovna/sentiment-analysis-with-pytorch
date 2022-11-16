import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext

from datasets import create_datasets
from utils import evaluate

# computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the last model checkpoint
model = torch.jit.load('outputs/model_scripted.pth')

# get the test dataset and the test data loader
train_data, test_data = create_datasets()

# load the vocabulary
vocab = torch.load('vocab.pth')

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

BATCH_SIZE = 32

def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(x): return int(x)

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list)
    text_list = torch.nn.utils.rnn.pad_sequence(
        text_list, batch_first=True, padding_value=0)
    batch = {'text': text_list,
             'labels': label_list}
    return batch

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             collate_fn=collate_batch)

# loss function
criterion = nn.CrossEntropyLoss().to(device)

test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
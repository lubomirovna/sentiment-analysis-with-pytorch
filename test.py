import torch
import torch.nn as nn

import re
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import CNN, reset_weights
from datasets import create_datasets
from utils import evaluate


def text_pipeline(text):
    """Processes the input text and returns a tensor representation."""
    tokens = re.findall(r"\b\w+\b", text)
    tensor = vocab(tokens)
    return tensor


# Load the vocabulary
vocab = torch.load('outputs/vocab.pth')

# Load pretrained embeddings
global_vectors = GloVe(name='twitter.27B', dim=200)

# Get the pre-trained embeddings for the words in the vocabulary
words = vocab.get_itos()
embeddings = global_vectors.get_vecs_by_tokens(words)


def collate_batch(data):
    inputs, labels = [], []
    for (_text, _label) in data:
        inputs.append(torch.tensor(text_pipeline(_text), dtype=torch.int64))
        labels.append(int(_label))

    # Pad sequential data to a max length of a batch
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.tensor(labels)

    return {
        'text': inputs,
        'labels': labels
    }


# Set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model hyperparameters
EMBEDDING_DIM = 200
N_FILTERS = 200
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 3
DROPOUT = 0.5

model = CNN(
    embeddings,
    True,
    None,
    EMBEDDING_DIM,
    N_FILTERS,
    FILTER_SIZES,
    OUTPUT_DIM,
    DROPOUT).to(device)

reset_weights(model)


# Load the best model
checkpoint = torch.load('outputs/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])


# Get the test dataset 
train_data, test_data = create_datasets('inputs/train.json')


BATCH_SIZE = 32

# Create a data loader for the test set
test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             collate_fn=collate_batch)

# Define the loss function
criterion = nn.CrossEntropyLoss().to(device)

# Evaluate the model on the test set
test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

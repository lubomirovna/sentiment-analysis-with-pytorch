import argparse
import sys
import os

import torchtext
import re

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np
from sklearn.model_selection import KFold

import time

from model import CNN, reset_weights
from datasets import create_datasets, build_vocab
from utils import train, evaluate, epoch_time, save_model, SaveBestModel



def create_arg_parser():
    """Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path",
                        help="Path to the file with training data.")
    parser.add_argument("-out", "--output_path",
                        help="Path where the model is saved to.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train the network for.")
    return parser


def text_pipeline(text):
    """Processes the input text and returns a tensor representation."""
    tokens = re.findall(r"\b\w+\b", text)
    tensor = vocab(tokens)
    return tensor


# Parse command-line arguments
arg_parser = create_arg_parser()
parsed_args = arg_parser.parse_args(sys.argv[1:])

if os.path.exists(parsed_args.input_path):
    input_path = parsed_args.input_path
if os.path.exists(parsed_args.output_path):
    output_path = parsed_args.output_path

# Import datasets
train_data, test_data = create_datasets(input_path)

# Extract the vocabulary from the training data
vocab = build_vocab(train_data, f'{output_path}vocab.pth')

# Load the pre-trained embeddings
global_vectors = torchtext.vocab.GloVe(name="twitter.27B", dim=200)

# Get the pre-trained embeddings for the words in the vocabulary
words = vocab.get_itos()
embeddings = global_vectors.get_vecs_by_tokens(words)


def collate_batch(data):
    inputs, labels = [], []
    for (_text, _label) in data:
        inputs.append(torch.tensor(text_pipeline(_text), 
                      dtype=torch.int64))
        labels.append(int(_label))

    # Pad sequential data to a max length of a batch
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = torch.tensor(labels)

    return {
        'text': inputs,
        'labels': labels
    }


# Define model hyperparameters
EMBEDDING_DIM = 200
N_FILTERS = 200
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 3
DROPOUT = 0.5

# Set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = CNN(
    embeddings,
    False,
    None,
    EMBEDDING_DIM,
    N_FILTERS,
    FILTER_SIZES,
    OUTPUT_DIM,
    DROPOUT).to(device)

reset_weights(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# Define learning parameters
N_FOLDS = 5
N_EPOCHS = parsed_args.epochs
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Create a save best model callback
save_best_model = SaveBestModel()

# Create a list of indices to use for cross-validation
splits = KFold(n_splits=N_FOLDS)

# Create a list to store the training and validation losses for each fold
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(
        splits.split(np.arange(len(train_data)))):
    
    print('Fold {}'.format(fold + 1))

    # Create data loaders for the training and validation sets
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  sampler=train_sampler,
                                  collate_fn=collate_batch)
    valid_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  sampler=valid_sampler,
                                  collate_fn=collate_batch)

    # Train the model for the given number of epochs
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_epoch_loss, train_epoch_acc = train(
            model, train_dataloader, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = evaluate(
            model, valid_dataloader, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_epoch_loss:.3f} | Train Acc: {train_epoch_acc*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_epoch_loss:.3f} |  Val. Acc: {valid_epoch_acc*100:.2f}%')

        # Check if the validation loss has stopped improving
        if epoch > 0 and valid_epoch_loss > np.min(valid_loss[:-1]):
            print("Validation loss stopped improving, interrupting training...")
            break

        # Save the model if it has the lowest validation loss so far
        save_best_model(
            valid_epoch_loss, epoch, model, optimizer, criterion, output_path
        )
        

# Save the latest model after training is completed
save_model(epoch, model, optimizer, criterion, fold, output_path)
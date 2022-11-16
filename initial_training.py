import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchtext
from torchtext.vocab import GloVe

import numpy as np
from sklearn.model_selection import KFold

import time
from model import CNN
from datasets import create_datasets, build_vocab
from utils import train, evaluate, epoch_time, save_model, SaveBestModel

import argparse
import sys
import os


OUTPUT_DIR = ""

def create_arg_parser():
    # Creates and returns the ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=20,
                    help='Number of epochs to train the network for.')
    parser.add_argument('-o', '--outputDirectory',
                    help='Path where the model is saved to.')
    return parser


arg_parser = create_arg_parser()
parsed_args = arg_parser.parse_args(sys.argv[1:])
if os.path.exists(parsed_args.outputDirectory):
    OUTPUT_DIR = parsed_args.outputDirectory


train_data, test_data = create_datasets()

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

vocab = build_vocab(train_data, "vocab.pth")

global_vectors = GloVe(name='twitter.27B', dim=200)
embeddings = global_vectors.get_vecs_by_tokens(vocab.get_itos())

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

# Build the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(vocab)
EMBEDDING_DIM = 200
N_FILTERS = 200
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 3
DROPOUT = 0.5

model = CNN(
    embeddings,
    False,
    INPUT_DIM,
    EMBEDDING_DIM,
    N_FILTERS,
    FILTER_SIZES,
    OUTPUT_DIM,
    DROPOUT).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

UNK_IDX = vocab['<unk>']
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

# Train the model

# learning parameters
N_FOLDS = 5
N_EPOCHS = parsed_args.epochs
BATCH_SIZE = 32

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#class weights for 3 class multi-class classification
class_weights = torch.tensor([0.5262, 0.9684, 0.5053]).to(device)
# loss function
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

# initialize SaveBestModel class
save_best_model = SaveBestModel()

splits = KFold(n_splits=N_FOLDS)

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

for fold, (train_idx, val_idx) in enumerate(
        splits.split(np.arange(len(train_data)))):

    print('Fold {}'.format(fold + 1))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_batch,
                                  sampler=train_sampler)
    valid_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_batch,
                                  sampler=valid_sampler)
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_epoch_loss, train_epoch_acc = train(
            model, train_dataloader, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = evaluate(model, valid_dataloader, criterion, device)
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

        # save the best model till now if it is the least loss in the current epoch
        save_best_model(
            valid_epoch_loss, epoch, model, optimizer, criterion, OUTPUT_DIR
        )

        # save the latest model after epoch is completed
        save_model(epoch, model, optimizer, criterion, fold, OUTPUT_DIR)
        
 
import pandas as pd
import text_preprocessing

import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import torch
import torchtext
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler

import model
import torch.nn as nn
import torch.optim as optim

import time

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('twitter_sentiment_data.csv')

# print(df.info())

# Label description

label = [-1, 0, 1, 2]

"""
    labelN = ["Anti", "Neutral", "Pro", "News"]
    labelDesc = [
        "the tweet does not believe in man-made climate change"
        , "the tweet neither supports nor refutes the belief of man-made climate change"
        , "the tweet supports the belief of man-made climate change"
        , "the tweet links to factual news about climate change"
    ]
"""

# Distribution of Sentiments

# plt.figure(figsize = (7, 7))
# plt.pie(df.sentiment.value_counts().values, labels = df.sentiment.value_counts().index, autopct = '%2.1f%%', textprops={'fontsize': 15})
# plt.title('Sentiment Distribution of the Tweet Dataset', fontsize=20)
# plt.tight_layout()
# plt.show()


# Data Preparation for Sentiment Analysis

df["message"] = df["message"].apply(text_preprocessing.preprocess_text)
df['sentiment'] = df['sentiment'].apply(lambda x: x+1)

print(df.head(12))

# Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(df['message'].tolist(),
                                                      df['sentiment'].tolist(),
                                                      test_size=0.3,
                                                      stratify = df['sentiment'].tolist(),
                                                      random_state=0)

train_data = list(zip(X_train, Y_train))
test_data = list(zip(X_test, Y_test))


# Build the vocab
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

min_freq = 3
special_tokens = ['<unk>', '<pad>']

vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_data),
                                                  min_freq=min_freq,
                                                  specials=special_tokens,
                                                  )
unk_index = vocab['<unk>']

vocab.set_default_index(unk_index)

# saving vectors for each word found in the dataset
torch.save(vocab, 'vocab.pth')

print(len(vocab))
print(vocab.get_itos()[:10])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) 

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         
    label_list = torch.tensor(label_list)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    batch = {'text': text_list,
             'labels': label_list}
    return batch

# Build the model

INPUT_DIM = len(vocab)
EMBEDDING_DIM = 300
N_FILTERS = 256
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = len(label)
DROPOUT = 0.2

model = model.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
                             
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

UNK_IDX = vocab['<unk>']
PAD_IDX = vocab['<pad>']

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# Train the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, dataloader, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()

    for batch in dataloader:

        tokens = batch['text'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        predictions = model(tokens)

        loss = criterion(predictions, labels)

        acc = categorical_accuracy(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len (dataloader)

def evaluate(model, dataloader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()

    with torch.no_grad():

        for batch in dataloader:

            tokens = batch['text'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(tokens)

            loss = criterion(predictions, labels)

            acc = categorical_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len (dataloader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

model = model.to(device)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters())

dataset = ConcatDataset([train_data, test_data])

N_FOLDS = 8
N_EPOCHS = 5
BATCH_SIZE = 64

splits = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

best_valid_loss = float('inf')

model_scripted = torch.jit.script(model)

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_data)))):
    
    print('Fold {}'.format(fold+1))

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

        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_scripted.save('CNNmodel.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

test_dataloader = DataLoader(test_data,
                              batch_size=BATCH_SIZE,
                              collate_fn=collate_batch)

test_loss, test_acc = evaluate(model, test_dataloader, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
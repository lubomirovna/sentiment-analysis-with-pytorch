import pandas as pd
import torch
import torchtext
from sklearn.model_selection import train_test_split

df = pd.read_csv('inputs/training_tweets.csv')


def create_datasets():
    """
    Function to build the training, and test dataset.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(df['tweet'].tolist(),
                                                        df['polarity'].tolist(),
                                                        test_size=0.3,
                                                        stratify=df['polarity'].tolist(),
                                                        random_state=42)
    train_data = list(zip(X_train, Y_train))
    test_data = list(zip(X_test, Y_test))
    print(f"Total training data: {len(train_data)}")
    print(f"Total validation data: {len(test_data)}")
    return train_data, test_data


tokenizer = torchtext.data.utils.get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


def build_vocab(train_data, path=None):
    """
    Function to build the vocab from training data.
    """
    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_data),
                                                      min_freq=3,
                                                      specials=["<unk>"])
    
    unk_index = vocab['<unk>']
    vocab.set_default_index(unk_index)

    if (path):
        torch.save(vocab, path)

    return vocab

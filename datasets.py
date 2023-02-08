import pandas as pd
import torch
import torchtext
from sklearn.model_selection import train_test_split


def create_datasets(path):
    """
    Function to load text data, build the training and test dataset.
    """
    df = pd.read_json(path, orient="records", lines=True)

    X_train, X_test, Y_train, Y_test = train_test_split(df['tweet'].tolist(),
                                                        df['sentiment'].tolist(),
                                                        test_size=0.3,
                                                        stratify=df['sentiment'].tolist(),
                                                        random_state=42)
    train_data = list(zip(X_train, Y_train))
    test_data = list(zip(X_test, Y_test))
    print(f"Total training data: {len(train_data)}")
    print(f"Total validation data: {len(test_data)}")
    return train_data, test_data


tokenizer = torchtext.data.utils.get_tokenizer('basic_english')


def yield_tokens(data_iter):
    """
    Function to tokenize input text and yield iterator of tokens.
    """
    for text, _ in data_iter:
        yield text.split()


def build_vocab(train_data, path):
    """
    Function to build the vocab from training data.
    """
    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_data),
                                                      min_freq=3,
                                                      specials=["<unk>"])

    unk_index = vocab['<unk>']
    vocab.set_default_index(unk_index)

    
    torch.save(vocab, path)

    return vocab

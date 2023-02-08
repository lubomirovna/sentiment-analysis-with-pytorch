import argparse
import sys
import os

import pandas as pd

import torch
from model import CNN

import torchtext
import re



def create_arg_parser():
    """Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the input file.")
    parser.add_argument("-out", "--output_path", help="Path where the output file is saved to.")
    parser.add_argument("-vocab", "--vocab_path", help="Path to the vocabulary.")
    parser.add_argument("-model", "--model_path", help="Path to the model to be used for the analysis.")
    return parser


def text_pipeline(text):
    """Processes the input text and returns a tensor representation."""
    tokens = re.findall(r"\b\w+\b", text)
    tensor = vocab(tokens)
    return tensor


def pad_tensor(tensor):
    """Pads the input tensor to the minimum length required by the model."""
    padding = min_len - tensor.size()[0]
    tensor = torch.nn.functional.pad(tensor, (0, padding))
    return tensor


def predict_class(model, sentence, text_pipeline=text_pipeline):
    """Predicts the sentiment class for the input sentence using the provided model."""
    model.eval()
    tensor = torch.tensor(text_pipeline(sentence), dtype=torch.int64).to(device)
    tensor = pad_tensor(tensor)
    tensor = tensor.unsqueeze(-2)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1)
    return max_preds.item()



# Parse command-line arguments
arg_parser = create_arg_parser()
parsed_args = arg_parser.parse_args(sys.argv[1:])

if os.path.exists(parsed_args.input_path):
    input_path = parsed_args.input_path
if os.path.exists(parsed_args.vocab_path):
    vocab_path = parsed_args.vocab_path
if os.path.exists(parsed_args.model_path):
    model_path = parsed_args.model_path
output_path = parsed_args.output_path


# Import datasets
tweets_df = pd.read_json(input_path, orient='records', lines=True)


# Load the vocabulary
vocab = torch.load(vocab_path)

# Load the pre-trained embeddings 
global_vectors = torchtext.vocab.GloVe(name='twitter.27B', dim=200)

# Extract the vocabulary from the pre-trained embeddings
words = vocab.get_itos()
embeddings = global_vectors.get_vecs_by_tokens(words)


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


# Load the best model
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Minimum sentence length required by the model
min_len = 5



# Mapping of class indices to sentiment labels
sentiment_label = {0: "Negative",
                   1: "Neutral",
                   2: "Positive"}

# Predict sentiment for each tweet
tweets_df['sentiment'] = tweets_df['processed_tweet'].apply(lambda str: sentiment_label[predict_class(model, str)])

print(tweets_df.sentiment.head())

# Store results to the json file
tweets_df.to_json(output_path, orient='records', lines=True)
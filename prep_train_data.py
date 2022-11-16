import argparse
import sys
import os
import pandas as pd
import numpy as np
from torch import float16
from text_preprocessing import preprocess_text

from sklearn.model_selection import train_test_split

INPUT_DIR = ""
OUTPUT_DIR = ""


def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser()
    parser.add_argument('inputDirectory',
                        help='Path to the input directory.')
    parser.add_argument('-o', '--outputDirectory',
                        help='Path to the output that contains the resumes.')
    return parser


arg_parser = create_arg_parser()
parsed_args = arg_parser.parse_args(sys.argv[1:])
if os.path.exists(parsed_args.inputDirectory):
    INPUT_DIR = parsed_args.inputDirectory
if os.path.exists(parsed_args.outputDirectory):
    OUTPUT_DIR = parsed_args.inputDirectory


# Import datasets

twitter_dataset = pd.read_csv(
    f'{INPUT_DIR}/140sentiment.csv',
    encoding="Latin-1",
    names=["polarity", "id", "date", "query", "user", "tweet"])

twitter_reddit_dataset = pd.read_csv(f'{INPUT_DIR}/Twitter_Data.csv')

# Remove unnecessary columns
twitter_dataset = twitter_dataset[["polarity", "tweet"]]
twitter_reddit_dataset = twitter_reddit_dataset[["category", "clean_text"]]

# Rename columns

twitter_reddit_dataset.columns = ["polarity", "tweet"]

# Setting up the labels. 0 negative, 1 neutral, 2 positive

twitter_dataset["polarity"] = twitter_dataset["polarity"].replace(2, 1)
twitter_dataset["polarity"] = twitter_dataset["polarity"].replace(4, 2)

twitter_reddit_dataset["polarity"] = twitter_reddit_dataset["polarity"].apply(
    lambda x: x + 1)

# Remove missing values

twitter_dataset.dropna(inplace=True)
twitter_reddit_dataset.dropna(inplace=True)

# Concat datasets

tweets_df = pd.concat([twitter_dataset, twitter_reddit_dataset], axis=0)

# Additional features
tweets_df["tweet"] = tweets_df["tweet"].astype(str)
tweets_df["polarity"] = tweets_df["polarity"].astype(np.float32)

tweets_df.reset_index(drop=True, inplace=True)


# Pre-processing
tweets_df['tweet'] = tweets_df['tweet'].apply(preprocess_text)


# Drop empty
tweets_df = tweets_df[~(tweets_df["tweet"] == '')]


# Length
tweets_df["length"] = tweets_df["tweet"].apply(len)


# Drop short sentences
tweets_df = tweets_df[~(tweets_df["length"] < 5)]

print(tweets_df.groupby("polarity")["length"].describe())

# Save data to a csv file
tweets_df.to_csv(f'{OUTPUT_DIR}/training_tweets.csv',
                 columns=['polarity', 'tweet'])

import pandas as pd
import text_preprocessing

import torch
import torchtext
import model

tweets = pd.read_csv('tweets.csv')
tweets['processed_tweet'] = tweets['tweet'].apply(text_preprocessing.preprocess_text)

print(tweets.head())

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

vocab = torch.load('vocab.pth')

text_pipeline = lambda x: vocab(tokenizer(x))

model = torch.jit.load('CNNmodel.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Predict sentiment 
min_len = 5

def pad_tensor(t):
    
     padding = min_len - t.size()[0]
     t = torch.nn.functional.pad(t, (0, padding))
     
     return t

def predict_class(model, sentence, text_pipeline=text_pipeline):
    model.eval()
    tensor = torch.tensor(text_pipeline(sentence)).to(device)
    tensor = pad_tensor(tensor)
    tensor = tensor.unsqueeze(-2)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    
    return max_preds.item()

sentiment_label = {0:"Anti",
                   1: "Neutral",
                   2: "Pro",
                   3: "News"}

processedTweets = []

for tweet in tweets['processed_tweet']:
    processedTweets.append([tweet, sentiment_label[predict_class(model, tweet)]])

# Creating a dataframe from the tweets list above
df = pd.DataFrame(processedTweets, columns=['tweet', 'sentiment'])

print(df.head())
   
df.to_csv('processedTweets.csv') 
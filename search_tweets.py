import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "#coronavirus OR #pandemic OR #pandemia OR covid since:2022-01-01 until:2023-01-01 lang:en -is:retweet -has:media" # Building a query
tweets = [] # Creating list to append tweet data 
limit = 100000

# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):

    if i == limit:
        break
    else:
        tweets.append([tweet.date, tweet.content, tweet.hashtags])
        # print(dir(tweet))


# Creating a dataframe from the tweets list above
df = pd.DataFrame(tweets, columns=['date', 'tweet','hashtags'])

df.to_json('data/Covid/c22.json', orient='records', lines=True) 
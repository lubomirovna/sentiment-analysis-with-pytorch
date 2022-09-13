import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "(climatechange globalwarming) lang:en until:2015-01-01 since:2010-01-01 -is:retweet" # Building a query
tweets = [] # Creating list to append tweet data 
limit = 20000 

# Using TwitterSearchScraper to scrape data and append tweets to list
for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])

# Creating a dataframe from the tweets list above
df = pd.DataFrame(tweets, columns=['date', 'user', 'tweet'])

df.to_csv('tweets.csv') 
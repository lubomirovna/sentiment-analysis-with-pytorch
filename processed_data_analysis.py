import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('processedTweets.csv')

plt.figure(figsize = (7, 7))
plt.pie(df.sentiment.value_counts().values, labels = df.sentiment.value_counts().index, autopct = '%2.1f%%', textprops={'fontsize': 15})
plt.title('Sentiment Distribution of the Processed Data', fontsize=20)
plt.tight_layout()
plt.show()
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:\\Users\\S.Bharathi\\Downloads\\Tweets.csv\\Tweets.csv")

# Display the first few rows of the dataset
print(df.head())

# Check the column names and data types
print(df.info())

# Assuming the tweets are in a column named 'text'
sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to the 'text' column and count sentiments
df['sentiment'] = df['text'].apply(get_sentiment)
sentiment_counts = df['sentiment'].value_counts()

# Print the sentiment counts
print(sentiment_counts)

# Plot the sentiment counts
labels = sentiment_counts.index
counts = sentiment_counts.values

plt.bar(labels, counts, color=['green', 'red', 'blue'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis of Tweets')
plt.show()

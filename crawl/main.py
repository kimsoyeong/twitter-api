import csv

from twitter_api import (
    API_KEY,
    API_KEY_SECRET,
    ACCESS_TOKEN,
    ACCESS_TOKEN_SECRET,
)

from cleaner import *
import twitter

twitter_api = twitter.Api(consumer_key=API_KEY,
                          consumer_secret=API_KEY_SECRET,
                          access_token_key=ACCESS_TOKEN,
                          access_token_secret=ACCESS_TOKEN_SECRET)

if __name__ == "__main__":
    queries = ["🔪 -🍎", "☠ OR 💀", "🙈", "🤬", "💩",
               "#BlackLivesMatter OR #BLM",
               "#hatechinese OR #worldhatechina",
               "anti black", "anti muslim"]
    base_query = "-is:quote lang:en"

    csvFile = open("./tweets-new-1.csv", "a")
    csvWriter = csv.writer(csvFile)

    csvWriter.writerow(['tweet', 'label', 'hate_words', 'emojis'])

    for q in queries:
        search = twitter_api.GetSearch(term=f'{base_query} {q}', count=100)

        # collect data
        tweets = []
        for tweet in search:
            tmp = re.sub("\n", "", tweet.text.strip())
            tmp = re.sub(",", " ", tmp)
            tweets.append(tmp)

        TC = TweetCleaner()

        for tweet in tweets:
            cleaned_tweet, hates, emojis = TC.clean_tweet(tweet)
            if len(hates) > 0:
                csvWriter.writerow([cleaned_tweet, 1, hates, emojis])
            else:
                csvWriter.writerow([cleaned_tweet, 0, hates, emojis])

    csvFile.close()
    print("Collection Done")

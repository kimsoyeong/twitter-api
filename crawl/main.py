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

query = "-is:quote lang:en (nigga OR #BlackLivesMatter OR paki OR ching)"

# #Atlantaprotest #BLM #ChangeTheSystem #JusticeForGeorgeFloyd #BlueLivesMatter
search = twitter_api.GetSearch(term=query, count=200)

# collect data
tweets = []
for tweet in search:
    # print(tweet.text.strip())
    tmp = re.sub("\n", "", tweet.text.strip())
    tmp = re.sub(",", " ", tmp)
    tweets.append(tmp)

TC = TweetCleaner()
csvFile = open("./tweets.csv", "a")
csvWriter = csv.writer(csvFile)

csvWriter.writerow(['tweet', 'label', 'hate_words', 'emojis'])
for tweet in tweets:
    ct, hashtags, emojis = TC.clean_tweet(tweet)
    if "*" in ct:
        csvWriter.writerow([tweet, 1, hashtags, emojis])
    else:
        csvWriter.writerow([tweet, 0, hashtags, emojis])

csvFile.close()
print("Collection Done")

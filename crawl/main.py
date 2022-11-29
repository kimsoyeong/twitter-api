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

query = "-is:quote lang:en (fuck OR #BlackLivesMatter)"

# #Atlantaprotest #BLM #ChangeTheSystem #JusticeForGeorgeFloyd #BlueLivesMatter
search = twitter_api.GetSearch(term=query, count=50)

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

for tweet in tweets:
    ct, hashtags = TC.clean_tweet(tweet)
    if "*" in ct:
        csvWriter.writerow([tweet, 1, hashtags])
    else:
        csvWriter.writerow([tweet, 0, hashtags])

csvFile.close()
print("Done")

import ast
import csv
import time

from twitter_api import (
    API_KEY,
    API_KEY_SECRET,
    BEARER_TOKEN,
    ACCESS_TOKEN,
    ACCESS_TOKEN_SECRET,
)

import tweepy
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)


class TwitterStream(tweepy.StreamingClient):
    def __init__(self, bearer_token, time_limit=15):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.data = []
        super(TwitterStream, self).__init__(bearer_token)

    def on_data(self, raw_data):
        if (time.time() - self.start_time) < self.time_limit:
            print(time.time() - self.start_time)
            self.data.append(ast.literal_eval(raw_data.decode('utf-8'))['data']['text'])
            return True
        else:
            self.running = False

    def on_errors(self, errors):
        if errors == 420:
            return False
        else:
            self.filter()
            return False

def delete_all_rules(rules):
    # 규칙 값이 없는 경우 None 으로 들어온다.
    if rules is None or rules.data is None:
        return None
    stream_rules = rules.data
    ids = list(map(lambda rule: rule.id, stream_rules))
    client.delete_rules(ids=ids)


client = TwitterStream(BEARER_TOKEN)
rules = client.get_rules()
delete_all_rules(rules)
client.add_rules(tweepy.StreamRule(value="-is:retweet -is:quote -has:media lang:en (☠)"))
client.filter()

csvFile = open("tweets.csv", "a")
csvWriter = csv.writer(csvFile)
for data in client.data:
    csvWriter.writerow([data.replace("\n", "")])
csvFile.close()

print(len(client.data))
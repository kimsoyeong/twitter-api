import re
from profanity_filter import ProfanityFilter


class TweetCleaner:
    def __init__(self):
        self.pf = ProfanityFilter(languages=['en'])

    def clean_tweet(self, tweet):
        hashtags = re.findall("#[A-Za-z0-9_]+", tweet)

        s = re.sub("@[A-Za-z0-9_]+", "", tweet)  # remove usertags
        s = re.sub("#[A-Za-z0-9_]+", "", s)  # remove hashtags

        if "RT : " in s:
            s = re.sub("RT : ", "", s)
        else:
            s = re.sub("RT", "", s)

        s = re.sub(r'http\S+', '', s)  # remove url in tweet
        s = re.sub('[()!?]', ' ', s)
        s = re.sub('\[.*?\]', ' ', s)
        # s = re.sub("[^a-z0-9]", " ", s) # remove emojis

        cs = self.pf.censor(s)  # check hate words
        cs = cs.split()
        s = s.split()
        hates = []
        for i in range(len(cs)):
            if '*' in cs[i]:
                hates.append(s[i])

        stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
        cs = [w.strip() for w in cs if not w in stopwords]

        return " ".join(word for word in cs), hates

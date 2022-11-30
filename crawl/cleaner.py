import re
from profanity_filter import ProfanityFilter


class TweetCleaner:
    def __init__(self):
        # You need to install 'en' package to use ProfanityFilter
        # put "venv/bin/python -m spacy download en" in your terminal
        self.stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
        self.pf = ProfanityFilter(languages=['en'])

    def clean_tweet(self, tweet):
        s = re.sub("@[A-Za-z0-9_]+", "", tweet)  # remove usertags
        s = re.sub("#[A-Za-z0-9_]+", "", s)  # remove hashtags

        if "RT : " in s:
            s = re.sub("RT : ", "", s)
        else:
            s = re.sub("RT", "", s)

        s = re.sub(r'http\S+', '', s)  # remove url in tweet
        s = re.sub('[()!?]', ' ', s)
        s = re.sub('\[.*?\]', ' ', s)
        emojis = re.findall("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", s)
        # s = re.sub("[^a-z0-9]", " ", s) # remove emojis

        cs = self.pf.censor(s)  # check hate words
        cs = cs.split()
        s = s.split()
        hates = []  # hate words in a tweet
        for i in range(len(cs)):
            if '*' in cs[i]:
                hates.append(s[i])

        cs = [w.strip() for w in cs if not w in self.stopwords]

        return " ".join(word for word in cs), hates, emojis

import re
from profanity_filter import ProfanityFilter


class TweetCleaner:
    def __init__(self):
        # You need to install 'en' package to use ProfanityFilter
        # put "venv/bin/python -m spacy download en" in your terminal
        self.stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
        self.pf = ProfanityFilter(languages=['en'])

    def clean_tweet(self, tweet):
        s = re.sub("@[A-Za-z0-9_]+", "@USER", tweet)  # remove usertags
        # s = re.sub("#[A-Za-z0-9_]+", "", s)  # remove hashtags

        if "RT : " in s:
            s = re.sub("RT : ", "", s)
        else:
            s = re.sub("RT", "", s)

        s = re.sub(r'http\S+', '', s)  # remove url in tweet
        s = re.sub('[()!?]', ' ', s)
        s = re.sub('\[.*?\]', ' ', s)

        emojis = re.findall("["
                            u"\U0001F600-\U0001F6FF"  # transport & map symbols
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F600-\U0001F6FF"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            # Missing emojis
                            u"\U0001F900-\U0001F9FF"
                            u"\U00002600-\U000026FF"
                            u"\U00002700-\U000027bf"
                            u"\U0001F191-\U0001F251"
                            u"\U0001F9D0-\U0001F9DF"
                            u"\U0001F926-\U0001F937"
                            u"\U0001F004-\U0001F0CF"
                            u"\U0001F170-\U0001F171"
                            u"\U0001F17E-\U0001F17F"
                            # u"\U0001F18E-\U00003030" # cant decode this emojis
                            u"\U00002B50-\U00002B55"
                            u"\U00002934-\U00002935"
                            u"\U00002B05-\U00002B07"
                            u"\U00002B1B-\U00002B1C"
                            u"\U00003297-\U00003299"
                            # u"\U0000303D-\U000000A9" # cant decode this emojis
                            # u"\U000000AE-\U00002122" # cant decode this emojis
                            u"\U000023F3-\U000024C2"
                            u"\U000023E9-\U000023EF"
                            # u"\U000025B6-\U000023F8" # cant decode this emojis
                            u"\U0001F4A6"
                            u"\U0001F5A5"
                            u"\U000026FD"
                            "🩸"
                            "]+", s)

# Define the regular expression


        # print(emojis)
        # s = re.sub("[^a-z0-9]", " ", s) # remove emojis

        cs = self.pf.censor(s)  # check hate words
        cs = cs.split()
        s = s.split()
        hates = []  # hate words in a tweet
        for i in range(len(cs)):
            if '*' in cs[i]:
                hates.append(s[i])

        cs = [w.strip() for w in cs if not w in self.stopwords]
        s = [w.strip() for w in s]
        return " ".join(s), hates, emojis

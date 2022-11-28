import numpy as np
import pandas as pd
import nltk
from nltk.tokenize.casual import TweetTokenizer

# read CSV
data = pd.read_csv("labeled_data.csv")

# exporting tweet column
tt = data['tweet'].tolist()
print('Tweet: ', tt)

t = TweetTokenizer()
for i in tt:
    # sentence = nltk.sent_tokenize(i)
    # dont tokenize emojis right  
    # word = nltk.word_tokenize(i)
    word = t.tokenize(i)
    print(word)
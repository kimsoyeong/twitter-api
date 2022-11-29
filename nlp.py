import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
import string

stop_word = set(stopwords.words('english'))

# read CSV
data = pd.read_csv("labeled_data.csv")
data.drop_duplicates()
# exporting tweet column
tt = data['tweet'].tolist()
print('Tweet: ', tt)

for i in range (len(tt)):
    tt[i] = re.sub(r"RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)", "", tt[i])

# print('Tweet: ', tt)
t = TweetTokenizer()
for i in tt:
    # sentence = nltk.sent_tokenize(i)
    # dont tokenize emojis right  
    # word = nltk.word_tokenize(i)
    word = t.tokenize(i)
    print(word)
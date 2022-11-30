import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_word = set(stopwords.words('english'))

# read CSV
data = pd.read_csv("labeled_data.csv")
data.drop_duplicates()
# exporting tweet column
tt = data['tweet'].tolist()
print('Tweet: ', tt)

for i in range(len(tt)):
    tt[i] = re.sub(r"RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)", "", tt[i])

# Lemmatizer
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
# Stemmer
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

for i in tt:
    # sentence = nltk.sent_tokenize(i)
    # dont tokenize emojis right
    # word = nltk.word_tokenize(i)
    words = tokenizer.tokenize(i)
    words = [word for word in words if word not in stop_word]
    lemma_word = [lemmatizer.lemmatize(word) for word in words]
    # stem_word_p = [porter_stemmer.stem(word) for word in words]
    # stem_word_l = [lancaster_stemmer.stem(word) for word in words]
    print(words)
    print(lemma_word)
    # print(stem_word_p)
    # print(stem_word_l)


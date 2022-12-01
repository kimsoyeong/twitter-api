from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer

# You need to download these things at first
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

stop_word = set(stopwords.words('english'))
stop_word_symbol = {"…", "’", ":", '"', '-', '️', '&', '“', '(', '/', "'", ";", "+", "*", "~"}
stop_word.update(stop_word_symbol)

# read CSV
data = pd.read_csv("./TweetBLM.csv")
test_data = pd.read_csv("./crawl/tweets-new.csv")
data.drop_duplicates(subset=['tweet'], inplace=True)
test_data.drop_duplicates(subset=['tweet'], inplace=True)

# exporting tweet column
tt = data['tweet'].tolist()
print('Number of Tweets: ', len(tt))

data['tweet'] = data['tweet'].str.replace("RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)",
                                          "")
test_data['tweet'] = test_data['tweet'].str.replace(
    "RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)",
    "")
# for i in range(len(tt)):
#     tt[i] = re.sub(r"RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)", "", tt[i])

# Lemmatizer
tokenizer = TweetTokenizer(reduce_len=True)
lemmatizer = WordNetLemmatizer()
# Tokenize
# data['tokenized'] = data['tweet'].apply(
#     lambda x: [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(x.lower()) if word not in stop_word])
# test_data['tokenized'] = test_data['tweet'].apply(
#     lambda x: [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(x.lower()) if word not in stop_word])

# Snowball stemmer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
data['tokenized'] = data['tweet'].apply(
    lambda x: [stemmer.stem(word) for word in tokenizer.tokenize(x.lower()) if word not in stop_word])
test_data['tokenized'] = test_data['tweet'].apply(
    lambda x: [stemmer.stem(word) for word in tokenizer.tokenize(x.lower()) if word not in stop_word])

print(data)

negative_words = np.hstack(data[data.label == 1]['tokenized'].values)
positive_words = np.hstack(data[data.label == 0]['tokenized'].values)
negative_word_count = Counter(negative_words)
positive_word_count = Counter(positive_words)
print(negative_word_count.most_common(20))
print(positive_word_count.most_common(20))
print()
print()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
text_len = data[data['label'] == 0]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Non-hate tweets')
ax1.set_xlabel('length of tweets')
ax1.set_ylabel('number of tweets')
print("Average length of Non-hate tweets: ", np.mean(text_len))

text_len = data[data['label'] == 1]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Hate tweets')
ax2.set_xlabel('length of tweets')
ax2.set_ylabel('number of tweets')
print("Average length of Hate tweets: ", np.mean(text_len))

# Integer Encoding
from tensorflow.keras.preprocessing.text import Tokenizer

X_data = data['tokenized'].values
Y_data = data['label'].values
X_test = test_data['tokenized'].values
Y_test = test_data['label'].values

tk = Tokenizer()
tk.fit_on_texts(X_data)

threshold = 2
total_cnt = len(tk.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tk.word_counts.items():
    total_freq = total_freq + value
    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('The size of vocabulary: ', total_cnt)
print('The number of rare words that appears less/equal than %s: %s' % (threshold - 1, rare_cnt))
print('The ratio of rare word in vocabulary:', (rare_cnt / total_cnt) * 100)
print('전체 등장 빈도에서 rare 단어 등장 빈도 비율:', (rare_freq / total_freq) * 100)

tk = Tokenizer(total_cnt, oov_token='OOV')
tk.fit_on_texts(X_data)
X_data = tk.texts_to_sequences(X_data)
X_test = tk.texts_to_sequences(X_test)

print('Max length of tweet: ', max(len(l) for l in X_data))
print('Avg length of tweet: ', sum(map(len, X_data)) / len(X_data))

ax3.hist([len(s) for s in X_data], bins=50)
ax3.set_xlabel('length of tweets')
ax3.set_ylabel('number of tweets')
plt.savefig('graph.png')

print()
print()


#################################
#                               #
#          MODEL PART           #
#                               #
#################################

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if len(s) <= max_len:
            cnt += 1
    print("The ratio of tweets whose length is less/equal than %s (in entire tweets): %s" % (
        max_len, (cnt / len(nested_list)) * 100))


max_len = 100
below_threshold_len(max_len, X_data)

from tensorflow.keras.preprocessing.sequence import pad_sequences

X_data = pad_sequences(X_data, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# GRU
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(total_cnt, 100))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model_GRU.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_data, Y_data, epochs=15, callbacks=[es, mc], batch_size=100, validation_split=0.2)

GRU_model = load_model('best_model_GRU.h5')
print("\n Test accuracy: %.4f" % (GRU_model.evaluate(X_test, Y_test)[1]))

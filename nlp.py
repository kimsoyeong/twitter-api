import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer

stop_word = set(stopwords.words('english'))
stop_word_symbol = {"…", "’", ":", '"', '-', '️', '&', '“', '(', '/', "'", ";", "+", "*", "~"}
stop_word.update(stop_word_symbol)

# read CSV
data = pd.read_csv("./TweetBLM.csv")
test_data = pd.read_csv("./crawl/tweets-new.csv")
data.drop_duplicates(subset=['tweet'], inplace=True)
test_data.drop_duplicates(subset=['tweet'], inplace=True)

# exporting tweet column
data['tweet'] = data['tweet'].str.replace("RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)", "")
test_data['tweet'] = test_data['tweet'].str.replace(
    "RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)",
    "")

# Lemmatizer
tokenizer = TweetTokenizer(reduce_len=True)
lemmatizer = WordNetLemmatizer()
# Tokenize
data['tokenized'] = data['tweet'].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(x.lower()) if word not in stop_word])
test_data['tokenized'] = test_data['tweet'].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(x.lower()) if word not in stop_word])

# Integer Encoding
from tensorflow.keras.preprocessing.text import Tokenizer

X_data = data['tokenized'].values
Y_data = data['label'].values
X_test = test_data['tokenized'].values
Y_test = test_data['label'].values

tk = Tokenizer()
tk.fit_on_texts(X_data)

total_cnt = len(tk.word_index)

tk = Tokenizer(total_cnt, oov_token='OOV')
tk.fit_on_texts(X_data)
X_data = tk.texts_to_sequences(X_data)
X_test = tk.texts_to_sequences(X_test)

#################################
#                               #
#          MODEL PART           #
#                               #
#################################

max_len = 100

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

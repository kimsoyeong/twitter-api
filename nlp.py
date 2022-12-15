#!/usr/bin/env python
# coding: utf-8

# !pip install tensorflow
# !pip install nltk
# !pip install scikit-learn


from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
from scipy.stats import pearsonr
import ast


class DataPreprocessor:
    def __init__(self):
        self.tokenizer = TweetTokenizer(reduce_len=True)
        self.lemmatizer = WordNetLemmatizer()
        
        self.regex = "RT (@[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|#|https\S+|http\S+|(?<!\d)[.,;:!?](?!\d)"
        self.emoji_dict = None
        self.stop_word = None
        
        self.tweet_vec = CountVectorizer(tokenizer=self.tokenize_lemmatize)
        self.emoji_vec = CountVectorizer()
        
        self.make_stop_word()
    
    def make_stop_word(self):
        self.stop_word = set(stopwords.words('english'))
        stop_word_symbol = {"…", "’", ":", '"', '-', '️', '&', '“', '(', '/', "'", ";", "+", "*", "~"}
        self.stop_word.update(stop_word_symbol)
    
    # def tokenize(self, text): # tokenize the tweets
    #     tknzr = TweetTokenizer()
    #     return tknzr.tokenize(text)

    def tokenize_lemmatize(self, text): # tokenize the tweets
        tknzr = TweetTokenizer()
        tokens= tknzr.tokenize(text)
        lemmatzr = WordNetLemmatizer()

        lemmas = [lemmatzr.lemmatize(token) for token in tokens if token not in self.stop_word]

        return [lemma.lower() for lemma in lemmas]
    
    def vectorize(self, data):      
        # Fit the vectorizer on the 'tweet' column
        self.tweet_vec.fit(data['tweet'])
        
        # Transform the 'tweet' column into a numerical representation
        tweet_vectors = self.tweet_vec.transform(data['tweet']) # matrix of token counts
        
        return tweet_vectors
            
    def vectorize_emojis(self, data):
        data['emojis'] = data['emojis'].apply(        
            lambda x: ast.literal_eval(x)
        )

        data['emojis'] = data['emojis'].apply(lambda x: (''.join(x) if len(x) > 0 else 'EMPTY')) # emoji가 꼭 있는 거로 data를 모아야 할 듯

        self.emoji_vec.fit(data['emojis']) # Fit the vectorizer to the 'emojis' column

        # Transform the 'emojis' column to a numerical representation
        emoji_vectors = self.emoji_vec.transform(data['emojis'])
        
        return emoji_vectors
    
    def preprocess(self, data):
        tweet_vectors = self.vectorize(data)
        emoji_vectors = self.vectorize_emojis(data)
        
        # concatenate the tweet vectors and emoji sequences into a single feature matrix
        combined_vec = np.hstack((tweet_vectors, emoji_vectors))
        return combined_vec # preprocessed data
    
dp = DataPreprocessor()

data = pd.read_csv('crawl/data.csv')
data['tweet'] = data['tweet'].str.replace(dp.regex, "")
data['tweet'] = data['tweet']

combined_vec = dp.preprocess(data)


from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from scipy.sparse import csr_matrix, hstack


# Base


target_data = data['label']

reg_coeff = 0.01

# model = Sequential()

# # Get the unique words in the input data
# combined_vec_matrix = hstack(combined_vec)
# input_data_array = combined_vec_matrix.toarray()
# X_train, X_test, y_train, y_test = train_test_split(input_data_array, target_data, test_size=0.2, random_state=42)

# # Further try
# # # Split the data into train and test sets
# # X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

# # # Split the train data into train and validation sets
# # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# unique_words = np.unique(X_train)

# vocab_size = len(unique_words) + 1

# # Set the total_cnt parameter in the Embedding layer
# model.add(Embedding(vocab_size, output_dim=100))
# model.add(GRU(128))
# model.add(Dense(1, activation='sigmoid',
#                 kernel_regularizer=regularizers.l1(reg_coeff), 
#                 bias_regularizer=regularizers.l2(reg_coeff)))

# es_l = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# es_a = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)
# mc = ModelCheckpoint('best_GRU.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# history = model.fit(X_train, y_train, 
#           batch_size=10,
#           epochs=20, 
#           verbose=1,
#           validation_split=0.2, 
#           callbacks=[es_l, es_a, mc])


# perf = model.evaluate(X_test, y_test)

# print('Test loss: %.4f' % perf[0])
# print('Test accuracy: %.2f' % (perf[1]*100))


# # KFold


# reg_coeff = 0.01

# model = Sequential()

# # Get the unique words in the input data
# combined_vec_matrix = hstack(combined_vec)

# non_zero_elements = combined_vec_matrix.count_nonzero()

# vocab_size = non_zero_elements + 1

# # Set the total_cnt parameter in the Embedding layer
# model.add(Embedding(vocab_size, output_dim=100))
# model.add(GRU(128))
# model.add(Dense(1, activation='sigmoid',
#                 kernel_regularizer=regularizers.l1(reg_coeff), 
#                 bias_regularizer=regularizers.l2(reg_coeff)))

# es_l = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# es_a = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4)
# mc = ModelCheckpoint('best_GRU.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


# # In[ ]:


# # Import the KFold class
# from sklearn.model_selection import KFold

# # Create a KFold object with 5 folds
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# X = combined_vec_matrix.toarray()
# y = target_data

# # Loop through the folds
# for train_index, val_index in kf.split(X):
#     # Get the training and validation data for this fold
#     X_train_fold, X_val_fold = X[train_index], X[val_index]
#     y_train_fold, y_val_fold = y[train_index], y[val_index]
    
#     # Compile and train the model
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.fit(X_train_fold, y_train_fold, 
#               batch_size=32, 
#               epochs=5, 
#               verbose=1,
#               validation_split=0.2,
#               callbacks=[es_l, es_a, mc])
    
#     # Evaluate the model on the validation data for this fold
#     val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
#     print(f'Fold val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}')


# # In[ ]:


# perf = model.evaluate(X_test, y_test)

# print('Test loss: %.4f' % perf[0])
# print('Test accuracy: %.2f' % (perf[1]*100))

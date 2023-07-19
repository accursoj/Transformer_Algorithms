import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # gets rid of harmless error messages
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open('FPL_Datasets/Chatbot/Simple_Chatbot/intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)


vocab_size = 1000
embedding_dim = 32 # original 16
max_len = 20
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

epochs = 550
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# saving model
model.save('FPL_Datasets/Chatbot/Simple_Chatbot/chat_model')

import pickle

# saving tokenizer
with open('FPL_Datasets/Chatbot/Simple_Chatbot/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# sving label encoder
with open('FPL_Datasets/Chatbot/Simple_Chatbot/label_encoder.pickle', 'wb') as ecn_file:
     pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)






import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
import string
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
import pickle


with open('intents.json') as content:
    data = json.load(content)

tags = []
inputs = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])


df = pd.DataFrame({"inputs": inputs, "tag": tags})


df['inputs'] = df['inputs'].apply(lambda wrd: ''.join([ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation]))


tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(df['inputs'])
X_train = pad_sequences(tokenizer.texts_to_sequences(df['inputs']))


le = LabelEncoder()
y_train = le.fit_transform(df['tag'])


with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle)
with open('label_encoder.pkl', 'wb') as handle:
    pickle.dump(le, handle)


vocabulary = len(tokenizer.word_index)
output_length = len(le.classes_)
input_shape = X_train.shape[1]

i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)

model = Model(i, x)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=200, verbose=1)


model.save("chatbot_model.h5")

print("Model training complete. Model saved as chatbot_model.h5")

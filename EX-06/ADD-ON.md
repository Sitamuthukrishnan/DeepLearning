Code:
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Sample data
sentences = [['I', 'love', 'NLP'], ['He', 'plays', 'football']]
tags = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Build vocabularies
word_vocab = sorted(set(word for sent in sentences for word in sent))
tag_vocab = sorted(set(tag for tag_seq in tags for tag in tag_seq))

word2idx = {word: i + 1 for i, word in enumerate(word_vocab)}  # +1 for padding index 0
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}

n_words = len(word2idx) + 1  # +1 for padding
n_tags = len(tag2idx)

# Convert words and tags to sequences
X = [[word2idx[word] for word in sent] for sent in sentences]
y = [[tag2idx[tag] for tag in tag_seq] for tag_seq in tags]

# Pad sequences
max_len = max(len(seq) for seq in X)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')

# Reshape labels for sparse_categorical_crossentropy
y = np.expand_dims(y, -1)

# Build model
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=n_words, output_dim=50)(input_layer)
bilstm = Bidirectional(LSTM(units=50, return_sequences=True))(embedding)
output = TimeDistributed(Dense(n_tags, activation='softmax'))(bilstm)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X, y, batch_size=2, epochs=10)

Output:
<img width="542" height="321" alt="Screenshot 2025-09-17 093019" src="https://github.com/user-attachments/assets/fa208916-4d0a-4ca6-8cab-a76bc03df61f" />

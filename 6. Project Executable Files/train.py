import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('20191226-reviews.csv')

# Check the columns in the DataFrame
print("Columns in the DataFrame:", data.columns)

# Preprocess your data
# Use 'body' for reviews and classify sentiment based on 'rating'
X = data['body'].astype(str).values  # Reviews
y = data['rating'].apply(lambda x: 1 if x >= 4 else 0).values  # Binary sentiment: 1 for positive, 0 for negative

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Save the model
model.save('model/sentiment_model.h5')

# Save the tokenizer for later use
import pickle
with open('model/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import matplotlib.pyplot as plt

# test comment

df = pd.read_csv('imdb_master.csv', encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)
padded = pad_sequences(sentences, maxlen=2000)  # Max review len turned out to be 2000

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.25, random_state=1000)

model = Sequential()
model.add(Embedding(2000, 50, input_length=2000))
model.add(Flatten())
model.add(layers.Dense(300, input_dim=2000, activation='relu'))  # fixed error: specified input_dim parameter
model.add(layers.Dense(3, activation='softmax'))  # fixed error: changed from sigmoid to softmax
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# plot accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and Validation Accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('# epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

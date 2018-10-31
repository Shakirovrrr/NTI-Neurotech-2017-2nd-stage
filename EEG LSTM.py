import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM, Embedding
import dataprep as dp

np.random.seed(8)

directory = 'files/'
files = ['state_0_day_3.csv', 'state_0_day_4.csv',
         'state_1_day_3.csv', 'state_1_day_4.csv',
         'state_2_day_3.csv', 'state_2_day_4.csv']

dataset = dp.load_files(files, dir=directory)

X_train, Y_train = dp.prepare_trainset(dataset, sec=121, feature=None)

model = Sequential([
	Embedding(input_dim=968, output_dim=128),
	LSTM(128, return_sequences=True),
	LSTM(64),
	Dense(32),
	Dropout(0.5),
	Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, shuffle=10)

predictset = dp.load_file('stest1.csv', dir=directory)
X_predict = dp.prepare_predictset(predictset, sec=121, feature=None)

print(model.predict(X_predict))

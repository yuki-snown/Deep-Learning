from keras.utils import np_utils
from keras.datasets import mnist

(x_trains, y_trains), (x_tests, y_tests) = mnist.load_data()

x_trains = x_trains.astype('float32')
x_trains /= 255
correct = 10
y_trains = np_utils.to_categorical(y_trains, correct)

x_tests = x_tests.astype('float32')
x_tests /= 255
y_tests = np_utils.to_categorical(y_tests, correct)

print(x_trains.shape)
print(x_tests.shape)

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.layers.recurrent import LSTM
from keras import optimizers, regularizers

model = Sequential()

model.add(InputLayer(input_shape=(28, 28)))
weight_decay = 1e-4
model.add(LSTM(units=128, dropout=0.25, return_sequences=True))
model.add(LSTM(units=128, dropout=0.25, return_sequences=True))
model.add(LSTM(units=128, dropout=0.25, return_sequences=False, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

model.summary()

history = model.fit(
    x_trains,
    y_trains,
    batch_size=100,
    epochs=10,
    verbose=1,
    validation_data=(x_tests, y_tests)
)
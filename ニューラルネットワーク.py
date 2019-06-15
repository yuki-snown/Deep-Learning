from keras.datasets import mnist
(x_trains, y_trains), (x_tests, y_tests) = mnist.load_data()

from keras.utils import np_utils

x_trains = x_trains.reshape(60000, 784)
x_trains = x_trains / 255

classes = 10

y_trains = np_utils.to_categorical(y_trains, classes)

x_tests = x_tests.reshape(10000, 784)
x_tests = x_tests / 255
y_tests = np_utils.to_categorical(y_tests, classes)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(20, input_dim=784, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()

history = model.fit(
    x_trains,
    y_trains,
    epochs=10,
    batch_size=100,
    verbose=1,
    validation_data=(x_tests, y_tests)
)
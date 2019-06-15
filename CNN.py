#プーリングあり
from keras.utils import np_utils
from keras.datasets import mnist

(x_trains, y_trains), (x_tests, y_tests) = mnist.load_data()

x_trains = x_trains.reshape(60000,28,28,1)
x_trains = x_trains.astype('float32')
x_trains /= 255
correct = 10
y_trains = np_utils.to_categorical(y_trains, correct)

x_tests = x_tests.reshape(10000,28,28,1)
x_tests = x_tests.astype('float32')
x_tests /= 255
y_tests = np_utils.to_categorical(y_tests, correct)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

model = Sequential()


#畳み込み１
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))

#畳みこみ２
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))

#プーリング１
model.add(MaxPooling2D(pool_size=(2,2)))

#畳み込み３
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))

#プーリング２
model.add(MaxPooling2D(pool_size=(2,2)))

#ドロップアウト
model.add(Dropout(0.5))

#Flatten
model.add(Flatten())

#全結合
model.add(Dense(128, activation='relu'))

#出力
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
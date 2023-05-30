import numpy as np
import tensorflow as tf
import pandas as pd


class SupervisedNeuralNetwork:
    model = None

    def __init__(self):
        df = pd.read_csv('../data.csv', delimiter=';')
        data = df.astype(float).to_numpy()  # Konwersja danych na typ float i zamiana na numpy array

        # Dane zostały wcześniej załadowane i przypisane do zmiennych x_train, y_train, x_test, y_test
        x_train, y_train, x_test, y_test = self.create_custom_dataset(data)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28,)),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=4, activation="linear")
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        model.summary()

        model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test), shuffle=True)
        self.model = model

    def create_custom_dataset(self, data, train_ratio=0.8):
        length = len(data)
        train_size = int(length * train_ratio)

        train_data = data[:train_size, :]
        test_data = data[train_size:, :]

        x_train = train_data[:, :28]
        y_train = train_data[:, 28:29]

        x_test = test_data[:, :28]
        y_test = test_data[:, 28:29]

        print(x_train.shape)
        print(y_train.shape)
        return x_train, y_train, x_test, y_test

    def predict(self,data):
        predictions = self.model.predict(data)
        print(np.max(predictions))
        return np.argmax(predictions)
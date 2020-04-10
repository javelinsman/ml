import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

def load_dataset():
    cifar10_builder = tfds.builder("cifar10")
    cifar10_builder.download_and_prepare()
    datasets = cifar10_builder.as_dataset()
    ds_train, ds_test = datasets['train'], datasets['test']

    train_images = np.array([data['image'].numpy() for data in ds_train])
    train_labels = np.array([data['label'] for data in ds_train])
    test_images = np.array([data['image'].numpy() for data in ds_test])
    test_labels = np.array([data['label'] for data in ds_test])

    train_images = train_images / 255
    test_images = test_images / 255

    return (train_images, train_labels), (test_images, test_labels)

class CNN:
    def __init__(self):
        self.model = self.__build_model()

    def __build_model(self):
        model = Sequential()
        model.add(Conv2D(4, 7, input_shape=(32, 32, 3)))
        model.add(MaxPool2D())
        model.add(Conv2D(8, 4))
        model.add(MaxPool2D())
        model.add(Conv2D(16, 2))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(units=10, activation='softmax'))
        return model

    def train(self, images, labels, epochs=30, batch_size=256):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        self.model.fit(images, labels,
                       epochs=epochs, batch_size=batch_size)

    def test(self, images, labels):
        predictions = self.model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        print('Accuracy:', (predicted_labels == labels).mean())
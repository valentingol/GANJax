import os

import cv2
import tensorflow as tf

def load_images_mnist(batch_size=128, seed=0):
    def prepare_dataset(X):
        X = tf.cast(X, tf.float32)
        # Normalization, pixels in [-1, 1]
        X = (X / 255.0) * 2.0 - 1.0
        # shape=(batch_size, 28, 28, 1)
        return X
    (X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
    X = tf.concat([X_train, X_test], axis=0)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.cache().shuffle(buffer_size=len(X), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=-1)
    dataset = dataset.map(prepare_dataset)
    return dataset


def load_images_cifar10(batch_size=128, seed=0):
    def prepare_dataset(X):
        X = tf.cast(X, tf.float32)
        # Normalization, pixels in [-1, 1]
        X = (X / 255.0) * 2.0 - 1.0
        # shape=(batch_size, 32, 32, 3)
        return X
    (X_train, _), (X_test, _) = tf.keras.datasets.cifar10.load_data()
    X = tf.concat([X_train, X_test], axis=0)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.cache().shuffle(buffer_size=len(X), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=-1)
    dataset = dataset.map(prepare_dataset)
    return dataset


def load_images_celeba_64(batch_size=128, seed=0, path='data/CelebA/images'):
    def generate_data():
        for f_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, f_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[20: -20, :, :]
            img = cv2.resize(img, (64, 64))
            img = tf.constant(img, dtype=tf.float32)
            img = (img / 255.0) * 2.0 - 1.0
            yield img
    dataset = tf.data.Dataset.from_generator(
        generate_data, output_types=tf.float32, output_shapes=(64, 64, 3)
        )
    dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=-1)
    dataset.__len__ = lambda : tf.constant(202_599 // batch_size + 1,
                                           dtype=tf.int64)
    return dataset


def load_images_celeba_128(batch_size=128, seed=0, path='data/CelebA/images'):
    def generate_data():
        for f_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, f_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[20: -20, :, :]
            img = cv2.resize(img, (128, 128))
            img = tf.constant(img, dtype=tf.float32)
            img = (img / 255.0) * 2.0 - 1.0
            yield img
    dataset = tf.data.Dataset.from_generator(
        generate_data, output_types=tf.float32, output_shapes=(128, 128, 3)
        )
    dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=-1)
    dataset.__len__ = lambda : tf.constant(202_599 // batch_size + 1,
                                           dtype=tf.int64)
    return dataset



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = load_images_celeba_128()
    tf.print('length:', ds.__len__())
    tf.print('spec:', ds.element_spec)
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(ds):
        x = (x + 1.0) / 2.0 # between [0, 1]
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.axis('off')
        plt.imshow(x[i, ...])
        if i == 24:
            break
    plt.show()

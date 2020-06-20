import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K


class Encoder(Model):

    def __init__(self):
        super(Encoder, self).__init__(name='encoder')

        self.conv1 = layers.Conv2D(
            16, (3, 3), activation='relu', padding='same')
        self.mp1 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = layers.Conv2D(
            8, (3, 3), activation='relu', padding='same')
        self.mp2 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv3 = layers.Conv2D(
            8, (3, 3), activation='relu', padding='same')
        self.mp3 = layers.MaxPooling2D((2, 2), padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128) #image embedding

    def call(self, inputs):
        x = self.mp1(self.conv1(inputs))
        # print("x.shape:", x.shape)
        x = self.mp2(self.conv2(x))
        x = self.mp3(self.conv3(x))
        # print("x.shape:", x.shape)
        outputs = self.fc1(self.flatten(x))
        # print("outputs.shape:", outputs.shape)
        return outputs


class Decoder(keras.Model):

    def __init__(self):
        super(Decoder, self).__init__(name='decoder')

        self.conv1 = layers.Conv2D(
            8, (3, 3), activation='relu', padding='same')
        self.us1 = layers.UpSampling2D((2, 2))
        self.conv2 = layers.Conv2D(
            8, (3, 3), activation='relu', padding='same')
        self.us2 = layers.UpSampling2D((2, 2))
        self.conv3 = layers.Conv2D(16, (3, 3), activation='relu')
        self.us3 = layers.UpSampling2D((2, 2))
        self.conv4 = layers.Conv2D(
            1, (3, 3), activation='sigmoid', padding='same')

    def call(self, inputs):
        x = K.reshape(inputs, [-1, 4, 4, 8])
        x = self.us1(self.conv1(x))
        x = self.us2(self.conv2(x))
        x = self.us3(self.conv3(x))
        outputs = self.conv4(x)
        return outputs


def main():

    e = Encoder()
    d = Decoder()

    x = tf.random.normal([2, 28, 28, 1])

    prob = e(x)
    print(prob)
    x_hat = d(prob)
    print(x_hat.shape)


if __name__ == '__main__':
    main()

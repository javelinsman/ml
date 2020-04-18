import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Embedding, Input
import tensorflow.keras.backend as K

from util.g2p import phoneme_types

D = {
    'latent': 77,
    'F': 128,
    'embedding': 64
}


class AudioEncoder:
    def __init__(self):
        self.conv1 = Conv1D(D['latent'], 1, activation='relu')
        self.conv2 = Conv1D(D['latent'], 1)
        self.model = self.__build()

    def __build(self):
        inputs = Input(shape=(None, D['F']))
        x = self.conv1(inputs)
        x = self.conv2(x)
        return Model(inputs, x)

class TextEncoder:
    def __init__(self):
        self.embedding = Embedding(len(phoneme_types), D['embedding'])
        self.conv1 = Conv1D(D['latent'], 1, activation='relu')
        self.conv_att = Conv1D(D['latent'], 1)
        self.conv_chr = Conv1D(D['latent'], 1)
        self.model = self.__build()

    def __build(self):
        inputs = Input(shape=(None,))
        embedded = self.embedding(inputs)
        x = self.conv1(embedded)
        encoded_att = self.conv_att(x)
        encoded_chr = self.conv_chr(x)
        return Model(inputs, [encoded_att, encoded_chr])


def mix_input(text_encoded_att, text_encoded_chr, audio_encoded):
    attention = K.batch_dot(text_encoded_att, K.permute_dimensions(audio_encoded, (0, 2, 1)))
    attention = K.softmax(attention / D['latent'] ** 0.5)
    mixed_input = K.batch_dot(K.permute_dimensions(attention, (0, 2, 1)), text_encoded_chr)
    input_to_decoder = K.concatenate([mixed_input, audio_encoded])
    return input_to_decoder, attention

class AudioDecoder:
    def __init__(self):
        self.conv1 = Conv1D(D['F'], 1, activation='relu')
        self.conv2 = Conv1D(D['F'], 1)
        self.model = self.__build()
    
    def __build(self):
        inputs = Input(shape=(None, 2 * D['latent']))
        x = self.conv1(inputs)
        x = self.conv2(x)
        return Model(inputs, x)

class TTSModel:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.audio_decoder = AudioDecoder()
        self.model = self.__build()

    def __build(self):
        text_input = Input(shape=(None,))
        audio_input = Input(shape=(None, D['F']))
        text_encoded_att, text_encoded_chr = self.text_encoder.model(text_input)
        audio_encoded = self.audio_encoder.model(audio_input)
        input_to_decoder, attention = \
            mix_input(text_encoded_att, text_encoded_chr, audio_encoded)
        self.attention = Model([audio_input, text_input], attention)
        audio_output = self.audio_decoder.model(input_to_decoder)
        model = Model([audio_input, text_input], audio_output)

        N = tf.shape(attention)[1]
        T = tf.shape(attention)[2]
        ts = K.reshape(tf.tile(tf.range(T), [N]), (N, T))
        ns = K.reshape(tf.repeat(tf.range(N), T), (N, T))
        attention_guide = tf.cast(1 - tf.exp(-(ns / N - ts / T) ** 2 / (2 * 0.2 ** 2)), float)
        attention_loss = K.mean(tf.math.multiply(attention, attention_guide))
        # model.add_loss(attention_loss)

        bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
        mse_loss_fn = tf.keras.losses.MeanSquaredError()

        def loss(y_pred, y_true):
            return mse_loss_fn(y_pred, y_true) ** 0.5 + bce_loss_fn(y_pred, y_true)

        model.compile(loss=loss, optimizer='adam')

        return model

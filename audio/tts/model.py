import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Embedding, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from util.g2p import phoneme_types
from util.layers import HighwayConv1D

D = {
    'latent': 256,
    'F': 80,
    'F_': 513,
    'c': 512,
    'embedding': 128
}

class AudioEncoder:
    def __init__(self):
        d = D['latent']
        self.layers = [
            Conv1D(d, 1, dilation_rate=1, activation='relu', padding="causal"),
            Conv1D(d, 1, dilation_rate=1, activation='relu', padding="causal"),
            Conv1D(d, 1, dilation_rate=1, padding="causal"),

            HighwayConv1D(3, 1),
            HighwayConv1D(3, 3),
            HighwayConv1D(3, 9),
            HighwayConv1D(3, 27),
            HighwayConv1D(3, 1),
            HighwayConv1D(3, 3),
            HighwayConv1D(3, 9),
            HighwayConv1D(3, 27),
            HighwayConv1D(3, 3),
            HighwayConv1D(3, 3),
        ]
        self.model = self.__build()

    def __build(self):
        inputs = Input(shape=(None, D['F']))
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return Model(inputs, x)

class TextEncoder:
    def __init__(self):
        e = D['embedding']
        d = D['latent']

        self.layers = [
            Embedding(len(phoneme_types), e),
            Conv1D(2 * d, 1, dilation_rate=1, activation='relu'),
            Conv1D(2 * d, 1, dilation_rate=1),

            HighwayConv1D(3, 1, padding='same'),
            HighwayConv1D(3, 3, padding='same'),
            HighwayConv1D(3, 9, padding='same'),
            HighwayConv1D(3, 27, padding='same'),
            HighwayConv1D(3, 1, padding='same'),
            HighwayConv1D(3, 3, padding='same'),
            HighwayConv1D(3, 9, padding='same'),
            HighwayConv1D(3, 27, padding='same'),
            HighwayConv1D(3, 1, padding='same'),
            HighwayConv1D(3, 1, padding='same'),

            HighwayConv1D(1, 1, padding='same'),
            HighwayConv1D(1, 1, padding='same'),
        ]
        self.model = self.__build()

    def __build(self):
        inputs = Input(shape=(None,))
        x = inputs
        for layer in self.layers:
            x = layer(x)
        encoded_att = x[:,:,:D['latent']]
        encoded_chr = x[:,:,D['latent']:]
        return Model(inputs, [encoded_att, encoded_chr])


def mix_input(text_encoded_att, text_encoded_chr, audio_encoded):
    attention = tf.matmul(text_encoded_att, audio_encoded, transpose_b=True)
    attention = K.softmax(attention / D['latent'] ** 0.5, axis=1)
    mixed_input = tf.matmul(attention, text_encoded_chr, transpose_a=True)
    input_to_decoder = K.concatenate([mixed_input, audio_encoded])
    # input_to_decoder = mixed_input
    return input_to_decoder, attention

class AudioDecoder:
    def __init__(self):
        d = D['latent']
        self.layers = [
            Conv1D(d, 1, dilation_rate=1, padding="causal"),
            HighwayConv1D(3, 1),
            HighwayConv1D(3, 3),
            HighwayConv1D(3, 9),
            HighwayConv1D(3, 27),
            HighwayConv1D(3, 1),
            HighwayConv1D(3, 1),
            Conv1D(d, 1, dilation_rate=1, activation='relu', padding="causal"),
            Conv1D(d, 1, dilation_rate=1, activation='relu', padding="causal"),
            Conv1D(d, 1, dilation_rate=1, activation='relu', padding="causal"),
            Conv1D(D['F'], 1, dilation_rate=1, activation='sigmoid', padding="causal"),
        ]
        self.model = self.__build()
    
    def __build(self):
        inputs = Input(shape=(None, 2 * D['latent']))
        # inputs = Input(shape=(None, D['latent']))
        x = inputs
        for layer in self.layers:
            x = layer(x)
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
        model.add_loss(attention_loss)

        bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
        l1_loss_fn = tf.keras.losses.MeanAbsoluteError()

        def loss(y_pred, y_true):
            return l1_loss_fn(y_pred, y_true) + bce_loss_fn(y_pred, y_true)

        optimizer = Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-6)

        model.compile(loss=loss, optimizer=optimizer)

        return model

class SSRN:
    # ignored deconv as I didn't reduced temporal dimension
    def __init__(self):
        self.layers = [
            Conv1D(D['c'], 1, dilation_rate=1),
            HighwayConv1D(3, 1),
            HighwayConv1D(3, 3),
            # TODO deconv 2*1
            HighwayConv1D(3, 1),
            HighwayConv1D(3, 3),
            # TODO deconv 2*1
            HighwayConv1D(3, 1),
            HighwayConv1D(3, 3),
            Conv1D(2 * D['c'], 1, dilation_rate=1),
            HighwayConv1D(3, 1),
            HighwayConv1D(3, 1),
            Conv1D(D['F_'], 1, dilation_rate=1),
            Conv1D(D['F_'], 1, dilation_rate=1, activation='relu'),
            Conv1D(D['F_'], 1, dilation_rate=1, activation='relu'),
            Conv1D(D['F_'], 1, dilation_rate=1, activation='sigmoid'),
        ]
        self.model = self.__build()
    
    def __build(self):
        inputs = Input(shape=(None, D['F']))
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return Model(inputs, x)


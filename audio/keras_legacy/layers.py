from tensorflow.keras.layers import Layer, Conv1D

class HighwayConv1D(Layer):
    def __init__(self, kernel_size=1, dilation_rate=1, activation='tanh', padding='causal'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.padding = padding

    def build(self, input_shape):
        self.d = input_shape[2]
        self.conv_h1 = Conv1D(self.d, self.kernel_size,
                              dilation_rate=self.dilation_rate,
                              activation=self.activation, padding=self.padding)
        self.conv_h2 = Conv1D(self.d, self.kernel_size,
                              dilation_rate=self.dilation_rate, padding=self.padding)
        super().build(input_shape)

    def call(self, x):
        sigH1 = self.conv_h1(x)
        H2 = self.conv_h2(x)
        return sigH1 * H2 + (1 - sigH1) * x

    def compute_output_shape(self, input_shape):
        return input_shape
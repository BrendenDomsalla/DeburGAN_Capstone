import tensorflow as tf
from tensorflow.keras import layers


class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Create trainable parameters for scale and offset
        self.gamma = self.add_weight(name='gamma', shape=(
            input_shape[-1],), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(
            input_shape[-1],), initializer='zeros', trainable=True)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # Compute mean and variance per instance (per sample in the batch)
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta


def residual_block(input_tensor, filters, use_dropout=False):
    """Builds a Residual Block with custom Instance Normalization."""
    x = layers.Conv2D(filters, kernel_size=3, strides=1,
                      padding="same")(input_tensor)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)
    if use_dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = InstanceNormalization()(x)
    return layers.add([input_tensor, x])

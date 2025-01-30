from tensorflow.keras import layers, Model
import sys
sys.path.append("C:\\Users\\bdoms\\DeburGAN_Capstone")

try:
    from util.Model_Functions import InstanceNormalization, residual_block
except IndexError:
    pass

# Define Residual Block


# Build Generator Network


def build_generator(input_shape=(720, 720, 3), num_res_blocks=9):
    """Builds the Generator network."""
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional block
    x = layers.Conv2D(64, kernel_size=7, strides=1, padding="same")(inputs)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Downsampling
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(num_res_blocks):
        x = residual_block(x, 256, use_dropout=True)

    # Upsampling
    x = layers.Conv2DTranspose(
        128, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer with global skip connection
    residual = layers.Conv2D(3, kernel_size=7, strides=1,
                             padding="same", activation="tanh")(x)
    outputs = layers.Add()([inputs, residual])

    return Model(inputs, outputs, name="Generator")

# Build Critic (Discriminator) Network


def build_critic(input_shape=(720, 720, 3)):
    """Builds the Critic (Discriminator) network."""
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters, stride):
        x = layers.Conv2D(filters, kernel_size=4,
                          strides=stride, padding="same")(x)
        x = InstanceNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        return x

    # PatchGAN-style architecture
    x = conv_block(inputs, 64, 2)  # No normalization on the first layer
    x = conv_block(x, 128, 2)
    x = conv_block(x, 256, 2)
    x = conv_block(x, 512, 1)

    # Output layer
    outputs = layers.Conv2D(1, kernel_size=4, strides=1, padding="same")(x)

    return Model(inputs, outputs, name="Critic")


# Instantiate models
generator = build_generator((256, 256, 3))
critic = build_critic((256, 256, 3))

# Summarize models
generator.summary()
critic.summary()

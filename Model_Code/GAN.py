# Define the VGG19 model for perceptual loss
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf
import time
from tensorflow import summary
import matplotlib.pyplot as plt
# import sys
# sys.path.append("C:\\Users\\bdoms\\DeburGAN_Capstone")

# try:
#     from util.Model_Functions import InstanceNormalization, residual_block
#     from util.DataLoader import Dataloader, Dataloader2
# except IndexError:
#     pass


def build_vgg19_layer(layer_name):
    """Creates a truncated VGG19 model up to the specified layer."""
    vgg = VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False
    output_layer = vgg.get_layer(layer_name).output
    model = Model(inputs=vgg.input, outputs=output_layer)
    return model


vgg19_layer = build_vgg19_layer("block3_conv3")


summary_writer = summary.create_file_writer(
    "C:\\Users\\bdoms\\DeburGAN_Capstone\\Checkpoints")


class GAN:
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, gen_optimizer=None, disc_optimizer=None, DataGenerator=None, shuffleData=True, crop=(720, 720), load_model=False) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.shuffleData = shuffleData

        self.step_counter = tf.Variable(0, dtype=tf.int64)
        self.DataGenerator = DataGenerator
        self.dataset = None
        self.lambda_gp = 10
        if gen_optimizer is None:
            self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        else:
            self.generator_optimizer = gen_optimizer

        if disc_optimizer is None:
            self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        else:
            self.discriminator_optimizer = disc_optimizer
        if load_model:
            self.load_model()

    def train(self, epochs) -> None:
        for epoch in range(epochs):
            self.dataset = self.DataGenerator.create_dataset()
            start = time.time()
            print(
                f"Epoch {epoch+1}/{epochs}, Folder {self.DataGenerator.folder_index.numpy()}")
            for image_batch in self.dataset:
                self.train_step(image_batch)
            if (self.DataGenerator.folder_index.numpy() + 1) % 5 == 0:
                self.save_model(epoch)
                self.showImages()
            secs = time.time() - start
            print(
                f'Time for epoch {epoch+1} is {int(secs//60)} mins, {secs%60} secs\nFolder Index: {self.DataGenerator.folder_index.numpy()}')

    @tf.function
    def train_step(self, batch) -> None:
        generator_input, real_images = batch
        for _ in range(5):  # Train discriminator multiple times per generator update
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(
                    generator_input, training=True)
                real_output = self.discriminator(real_images, training=True)
                fake_output = self.discriminator(
                    generated_images, training=True)
                # Compute Gradient Penalty
                gp = self.gradient_penalty(real_images, generated_images)
                disc_loss = self.discriminator_loss(
                    real_output, fake_output) + self.lambda_gp * gp
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(generator_input, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            # Compute adversarial loss and perceptual content loss
            adv_loss = self.adversarial_loss(fake_output)
            content_loss = self.perceptual_loss(real_images, generated_images)

            # Total generator loss is adversarial + content loss
            gen_loss = adv_loss + 100 * content_loss

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss,
                              step=self.step_counter)
            tf.summary.scalar("Discriminator Loss",
                              disc_loss, step=self.step_counter)
            self.step_counter.assign_add(1)

    def save_model(self, epoch):
        checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            folder_index=self.DataGenerator.folder_index
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory='C:\\Users\\bdoms\\DeburGAN_Capstone\\Checkpoints', max_to_keep=5
        )
        checkpoint_manager.save()
        print(
            f"Saved checkpoint for epoch {epoch + 1}, folder {self.DataGenerator.folder_index.numpy()}")

    def load_model(self):
        checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            folder_index=self.DataGenerator.folder_index
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=f'C:\\Users\\bdoms\\DeburGAN_Capstone\\Checkpoints', max_to_keep=5
        )
        if checkpoint_manager.latest_checkpoint:
            print(checkpoint_manager.latest_checkpoint)
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(
                f"Restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
            print(
                f"Folder Index restored to: {self.DataGenerator.folder_index.numpy()}")
        else:
            print("No checkpoint found. Starting from scratch.")

    def save_generator(self):
        self.generator.save(
            f'C:\\Users\\bdoms\\DeburGAN_Capstone\\Models\\generator.keras')

    def ModelSummary(self) -> None:
        self.generator.summary()
        self.discriminator.summary()

    @tf.function
    def adversarial_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    @tf.function
    def discriminator_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    @tf.function
    def perceptual_loss(self, real_images, generated_images):
        """Computes the perceptual loss using the VGG19 feature maps"""
        # Get feature maps from VGG19
        real_features = vgg19_layer(real_images)
        generated_features = vgg19_layer(generated_images)

        # Calculate the L2 loss between the feature maps
        return tf.reduce_mean(tf.square(real_features - generated_features))

    @tf.function
    def gradient_penalty(self, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_output = self.discriminator(
                interpolated, training=True)
        gradients = gp_tape.gradient(interpolated_output, [interpolated])[0]
        gradients_norm = tf.sqrt(tf.reduce_sum(
            tf.square(gradients), axis=[1, 2, 3]) + 1e-8)
        return tf.reduce_mean((gradients_norm - 1.0) ** 2)

    def display_images(self, image_batches, epoch=0, num_shown_images=3) -> None:
        num_batches = len(image_batches)
        plt.figure(figsize=(10, 10))  # Adjust figure size for better spacing
        for batch_idx in range(num_batches):
            images = image_batches[batch_idx]
            num_images = min(len(images), num_shown_images)
            for i in range(num_images):
                number = batch_idx * num_shown_images + i + 1
                denormed = self.__denormalize(images[i])
                # denormed=self.normalize_out_of_range_pixels(self.__denormalize(images[i]))
                plt.subplot(num_batches, num_shown_images, number)
                plt.imshow(denormed)  # Ensure images are in displayable format

                plt.axis('off')
        plt.suptitle(f'Epoch {epoch + 1}', fontsize=16)
        plt.tight_layout(pad=.5)
        plt.subplots_adjust(wspace=0.1, hspace=0.01)
        plt.show()

    def __denormalize(self, image):
        return (image + 1) / 2

    def showImages(self, num_images=3) -> None:
        for intake, out in self.dataset.take(1):
            generated_images = self.generator(intake, training=False)
            # kernel_sharpened = sharpen_image_batch(intake)
            # self.display_images((intake, generated_images, kernel_sharpened, out))
            self.display_images((intake, generated_images, out),
                                num_shown_images=num_images)

    @tf.function
    def __calc_accuracy(self, real_output, fake_output):
        real_accuracy = tf.reduce_mean(tf.cast(real_output >= 0.5, tf.float32))
        fake_accuracy = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
        return real_accuracy, fake_accuracy

    def show_latest_checkpoint(self, num_images=3):
        self.showImages(num_images=num_images)

    def normalize_out_of_range_pixels(self, image):
        # Define the valid range
        lower_bound = -0.5
        upper_bound = 1.5

        # Create a mask for pixels outside the valid range
        out_of_range_mask = tf.logical_or(image < 0.0, image > 1.0)

        # Normalize only the out-of-range pixels to [0, 1]
        normalized_image = tf.where(
            out_of_range_mask,
            # Normalize out-of-range pixels
            (image - lower_bound) / (upper_bound - lower_bound),
            image  # Keep in-range pixels unchanged
        )

        return normalized_image

    def create_dataset(self):
        self.dataset = self.DataGenerator.create_dataset()

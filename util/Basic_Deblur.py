import tensorflow as tf
import cv2
import numpy as np


def sharpen_image_batch(image_batch):
    # Define a simple sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Create an empty list to store sharpened images
    sharpened_images = []

    # Iterate over each image in the batch
    for image in image_batch:
        # Convert each image to a NumPy array
        image_np = image.numpy()
        # Ensure the image is in the range [0, 255] before applying the kernel
        image_np = (image_np + 1) * 127.5  # Rescale to [0, 255]
        # Apply sharpening kernel using OpenCV
        sharpened_image = cv2.filter2D(image_np, -1, kernel)

        # Normalize back to range [-1, 1]
        sharpened_image = (sharpened_image / 127.5) - 1

        # Convert back to TensorFlow eager tensor and append to list
        sharpened_images.append(tf.convert_to_tensor(
            sharpened_image, dtype=tf.float32))

    # Stack all sharpened images back into a batch
    sharpened_image_batch = tf.stack(sharpened_images)

    return sharpened_image_batch

# Example usage:
# Assuming 'image_batch' is a batch of images with shape (5, 256, 256, 3) and normalized to [-1, 1]
# sharpened_image_batch = sharpen_image_batch(image_batch)

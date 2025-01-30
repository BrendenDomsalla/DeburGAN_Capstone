import gradio as gr
import tensorflow as tf
import numpy as np
import os
from util.Model_Functions import InstanceNormalization


def load_model(save_path):
    """
    Loads a TensorFlow model from the specified path.
    :param save_path: Path to load the model from.
    :return: Loaded model.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"The specified path {save_path} does not exist.")
    model = tf.keras.models.load_model(save_path, custom_objects={
                                       "InstanceNormalization": InstanceNormalization})
    print(f"Model loaded from {save_path}")
    return model


model = load_model(r'C:\Users\bdoms\DeburGAN_Capstone\Models\generator.keras')


def deblur_image(input_image):
    try:
        # Ensure input is a valid NumPy array
        if not isinstance(input_image, np.ndarray):
            raise ValueError("Input is not a valid NumPy array")

        # Convert input image to tf.float32
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

        # Ensure input image has 3 channels
        if input_image.shape[-1] != 3:
            raise ValueError("Input image must have 3 color channels")

        # Normalize to [-1, 1] (as per model training)
        input_image = (input_image / 127.5) - 1

        # Determine dimensions for tiling
        height, width, _ = input_image.shape
        tile_size = 256

        # Calculate the maximum dimensions divisible by tile_size
        crop_height = (height // tile_size) * tile_size
        crop_width = (width // tile_size) * tile_size

        # Center-crop the image to the largest dimensions divisible by tile_size
        offset_height = (height - crop_height) // 2
        offset_width = (width - crop_width) // 2
        cropped_image = tf.image.crop_to_bounding_box(
            input_image, offset_height, offset_width, crop_height, crop_width
        )

        # Split the cropped image into tiles of size 256x256
        tiles = []
        for i in range(0, crop_height, tile_size):
            for j in range(0, crop_width, tile_size):
                tile = cropped_image[i:i+tile_size, j:j+tile_size, :]
                tiles.append(tile)

        # Process each tile with the model
        processed_tiles = []
        for tile in tiles:
            tile = tile[None, ...]  # Add batch dimension
            # Assuming model returns a batch of images
            output_tile = model(tile)[0]
            processed_tiles.append(output_tile)

        # Stitch the processed tiles back together
        rows = crop_height // tile_size
        cols = crop_width // tile_size
        stitched_image = tf.concat([
            tf.concat(processed_tiles[row * cols:(row + 1) * cols], axis=1)
            for row in range(rows)
        ], axis=0)

        # Denormalize the output back to [0, 1] for Gradio compatibility
        stitched_image = ((stitched_image + 1) / 2.0).numpy()

        # Clip values to ensure they are in the range [0, 1]
        stitched_image = np.clip(stitched_image, 0.0, 1.0)

        return stitched_image, "No Errors Detected"

    except Exception as e:
        error_msg = f"Error during image processing: {e}"
        print(error_msg)
        # Return a blank image and error message
        return np.zeros((256, 256, 3), dtype="float32"), error_msg


# Gradio interface
interface = gr.Interface(
    fn=deblur_image,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Text()],
    live=True,
    title="Image Deblurring AI",
    description="Upload a blurry image, and this AI will try to deblur it."
)

# Launch the interface
interface.launch(debug=True)

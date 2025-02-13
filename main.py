import gradio as gr
import tensorflow as tf
import numpy as np
import os
from util.Model_Functions import InstanceNormalization
import traceback


def load_model(save_path):
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"The specified path {save_path} does not exist.")
    model = tf.keras.models.load_model(save_path, custom_objects={
        "InstanceNormalization": InstanceNormalization})
    print(f"Model loaded from {save_path}")
    return model


model = load_model(r'C:\Users\bdoms\DeburGAN_Capstone\Models\generator.keras')


def deblur_image(image):
    try:
        image = (image.astype("float32")/127.5)-1
        image_size = 256
        overlap = 12
        dist = image_size-overlap
        height = (len(image)//dist)*dist
        width = (len(image[0])//dist)*dist

        processed_tiles = []
        for y in range(height//dist):
            row = []
            for x in range(width//dist):
                tile = image[y*dist:y*dist+image_size,
                             x*dist:x*dist+image_size]
                tile = np.expand_dims(tile, axis=0)
                tile = model(tile)[0]
                tile = tile[overlap//2:-overlap//2, overlap//2:-overlap//2]
                row.append(tile)
            processed_tiles.append(row)

        # Create the right edge tiles
        right_edges = []
        for i in range(height//dist):
            tile = image[i*dist:i*dist+image_size, -image_size:]
            tile = np.expand_dims(tile, axis=0)
            tile = model(tile)[0]
            tile = tile[overlap//2:-overlap//2, -(image.shape[1]-width):]
            right_edges.append(tile)
        right_edges = np.array(right_edges)

        # Bottom Tiles
        bottom_tiles = []
        for i in range(width//dist):
            tile = image[-image_size:, i*dist:i*dist+image_size]
            tile = np.expand_dims(tile, axis=0)
            tile = model(tile)[0]
            tile = tile[-(image.shape[0]-height):, overlap//2:-overlap//2]
            bottom_tiles.append(tile)

        bottom_right_tile = image[-image_size:, -image_size:]
        bottom_right_tile = np.expand_dims(bottom_right_tile, axis=0)
        bottom_right_tile = model(bottom_right_tile)[0]
        bottom_right_tile = bottom_right_tile[-(
            image.shape[0]-height):, -(image.shape[1]-width):]

        bottom_tiles = np.array(bottom_tiles)
        bottom_tiles = np.hstack(bottom_tiles)
        bottom_tiles = np.hstack([bottom_tiles, bottom_right_tile])

        processed_tiles = np.array(processed_tiles)
        # right_edges = np.expand_dims(right_edges, axis=0)
        right_edges = np.vstack(right_edges)
        output_image = [np.hstack(row) for row in processed_tiles]
        output_image = np.vstack(output_image)
        output_image = np.hstack(
            [output_image, right_edges])
        output_image = np.vstack([output_image, bottom_tiles])

        output_image = np.clip((output_image+1)/2, 0, 1)

        return output_image, "No Errors"
    except Exception as e:
        errorMsg = f"Something went wrong. {e}\n{traceback.format_exc()}"
        # raise Exception(errorMsg)
        return np.zeros((100, 100, 3)), errorMsg


# image = cv2.imread(
#     r"C:\Users\bdoms\DeburGAN_Capstone\Datasets\Train\blurred\012\00000000.png")
# im = deblur_image(image)
# cv2.imshow("image", im[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
interface = gr.Interface(
    fn=deblur_image,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Text()],
    live=True,
    title="Image Deblurring AI",
    description="Upload a blurry image, and this AI will try to deblur it."
)

interface.launch(debug=True)

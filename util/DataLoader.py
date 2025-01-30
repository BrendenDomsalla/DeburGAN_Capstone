from itertools import zip_longest
import tensorflow as tf
import os
import random


class DataLoader:
    def __init__(self, blurred_folder, sharp_folder, batch_size=10, crop=(720, 720)) -> None:
        self.blurred_folder = blurred_folder
        self.sharp_folder = sharp_folder
        self.batch_size = batch_size
        self.crop = crop

    def load_and_preprocess_image(self, blurred_path, sharp_path) -> tuple[tf.Tensor, tf.Tensor]:
        # Load sharp image
        sharp_image = tf.io.read_file(sharp_path)
        sharp_image = tf.image.decode_image(sharp_image, channels=3)
        sharp_image = tf.cast(sharp_image, tf.float32) / 127.5 - 1

        # Load blurred image
        blurred_image = tf.io.read_file(blurred_path)
        blurred_image = tf.image.decode_image(blurred_image, channels=3)
        blurred_image = (tf.cast(blurred_image, tf.float32)/127.5)-1

        crop_height, crop_width = self.crop

        # Get the dimensions of the images (assuming sharp_image and blurred_image have the same dimensions)
        image_shape = tf.shape(sharp_image)
        height, width = image_shape[0], image_shape[1]

        # Generate random offsets for cropping
        offset_height = tf.random.uniform(
            [], 0, height - crop_height + 1, dtype=tf.int32)
        offset_width = tf.random.uniform(
            [], 0, width - crop_width + 1, dtype=tf.int32)

        # Crop both images using the same offsets
        sharp_image = tf.image.crop_to_bounding_box(
            sharp_image, offset_height, offset_width, crop_height, crop_width)
        blurred_image = tf.image.crop_to_bounding_box(
            blurred_image, offset_height, offset_width, crop_height, crop_width)

        return blurred_image, sharp_image

    def create_dataset(self) -> tf.data.Dataset:

        # Get sorted lists of file paths to ensure matching order
        blurred_paths = sorted(tf.io.gfile.glob(
            os.path.join(self.blurred_folder, "*")))
        sharp_paths = sorted(tf.io.gfile.glob(
            os.path.join(self.sharp_folder, "*")))

        # Ensure the datasets are aligned
        if len(blurred_paths) != len(sharp_paths):

            raise ValueError(
                "Number of images in blurred and sharp folders must be the same.")

        # Shuffle paths together to ensure paired data remains aligned
        paired_paths = list(zip(blurred_paths, sharp_paths))
        random.shuffle(paired_paths)
        blurred_paths, sharp_paths = zip(*paired_paths)

        # Create a dataset of file path pairs
        file_pairs = tf.data.Dataset.from_tensor_slices(
            (list(blurred_paths), list(sharp_paths)))

        dataset = file_pairs.map(
            self.load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset


"""
Used for when the dataset folder is divided up into sub-folders, like in my new dataset
"""


class DataLoader2(DataLoader):
    def __init__(self, blurred_folder, sharp_folder, batch_size=3, crop=(720, 720), shuffle=True, num_folders=3, folder_index=0, total_folders=None) -> None:
        super().__init__(blurred_folder, sharp_folder, batch_size, crop)
        self.folder_index = tf.Variable(folder_index, dtype=tf.int64)
        self.num_folders = num_folders
        self.shuffle = shuffle
        if total_folders is None:
            self.total_folders = len(os.listdir(self.blurred_folder))
        else:
            self.total_folders = total_folders

    def create_dataset(self) -> tf.data.Dataset:
        blurred_paths = []
        sharp_paths = []

        for i in range(self.num_folders):
            blurred_folder_path = self.get_subfolder_by_index(
                self.blurred_folder, (self.folder_index.numpy()+i) % self.total_folders)
            sharp_folder_path = self.get_subfolder_by_index(
                self.sharp_folder, (self.folder_index.numpy()+i) % self.total_folders)
            # Add image paths from the current folder to the list
            blurred_paths.extend(sorted(tf.io.gfile.glob(
                os.path.join(blurred_folder_path, "*"))))
            sharp_paths.extend(sorted(tf.io.gfile.glob(
                os.path.join(sharp_folder_path, "*"))))

        self.folder_index.assign_add(self.num_folders)
        # Ensure the datasets are aligned
        if len(blurred_paths) != len(sharp_paths):

            blurred_paths, sharp_paths = self.fix_missing_files(
                blurred_paths, sharp_paths)
            if len(blurred_paths) != len(sharp_paths):
                print(len(blurred_paths), len(sharp_paths))
                for i in zip_longest(blurred_paths, sharp_paths):
                    print(i)
                raise ValueError(
                    f"Dataset is mismatched. Check folders from {self.get_subfolder_by_index(self.blurred_folder, self.folder_index.numpy())} "
                    f"and {self.get_subfolder_by_index(self.sharp_folder, self.folder_index.numpy())}."
                )

        # Shuffle paths together to ensure paired data remains aligned
        if self.shuffle:
            paired_paths = list(zip(blurred_paths, sharp_paths))
            random.shuffle(paired_paths)
            blurred_paths, sharp_paths = zip(*paired_paths)

        # Create a dataset of file path pairs
        file_pairs = tf.data.Dataset.from_tensor_slices(
            (list(blurred_paths), list(sharp_paths)))

        dataset = file_pairs.map(
            self.load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    def fix_missing_files(self, blurred, sharp):
        i = 0
        while i < len(blurred) and i < len(sharp):
            bf, sf = (blurred[i].replace(self.blurred_folder, ""),
                      sharp[i].replace(self.sharp_folder, ""))
            if (bf[1:4] != sf[1:4]):
                if bf[1:4] < sf[1:4]:
                    blurred.pop(i)
                else:
                    sharp.pop(i)
                continue

            if int(bf[5:-4]) < int(sf[5:-4]):
                blurred.pop(i)
                i -= 1
            elif int(bf[5:-4]) > int(sf[5:-4]):
                sharp.pop(i)
                i -= 1
            i += 1
        if len(blurred) > len(sharp):
            blurred.pop(-1)
        elif len(blurred) < len(sharp):
            sharp.pop(-1)
        return blurred, sharp

    @staticmethod
    def get_subfolder_by_index(parent_folder, index):
        # List all items in the parent folder
        all_items = os.listdir(parent_folder)

        # Filter to include only subfolders
        subfolders = [item for item in all_items if os.path.isdir(
            os.path.join(parent_folder, item))]

        # Sort the subfolders by name (convert to integers if names are numeric)
        subfolders.sort(key=lambda x: int(x) if x.isdigit() else x)

        # Get the subfolder by index
        if 0 <= index < len(subfolders):
            return os.path.join(parent_folder, subfolders[index])
        else:
            raise IndexError("Index out of range")

import numpy as np
import os.path as osp
import cv2


# read npz and save to png
def read_npz_and_save_png(npz_path, output_path):
    data = np.load(npz_path)
    # Assuming the npz file contains a single array named 'arr_0'
    img_array = data['arr_0']

    # Normalize the array to 0-255 if it's not already
    if img_array.dtype != np.uint8:
        img_array = (img_array - img_array.min()) / \
            (img_array.max() - img_array.min()) * 255
        img_array = img_array.astype(np.uint8)

    # transpose [1, 1080, 1920] to [1080, 1920, 1]
    if img_array.ndim == 3 and img_array.shape[0] == 1:
        img_array = img_array.transpose(1, 2, 0)

    print(np.unique(img_array, return_counts=True))

    # Save the array as a PNG image
    cv2.imwrite(output_path, img_array)


if __name__ == '__main__':
    data_dir = '/home/sinter/workspace/basic-learning/install/data/pw'

    read_npz_and_save_png(
        osp.join(data_dir, 'image_H42-G20240530-39-C-3483.npz'), osp.join(data_dir, 'image_output.png'))
    read_npz_and_save_png(
        osp.join(data_dir, 'anno_H42-G20240530-39-C-3483.npz'), osp.join(data_dir, 'anno_output.png'))

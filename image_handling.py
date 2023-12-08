import cv2
import numpy as np


def save_pixels_as_image(input_layer, image_shape, image_name):
    image_pixels = input_layer.reshape(image_shape)
    image_pixels = image_pixels.astype(np.uint8)
    cv2.imwrite(image_name, image_pixels)

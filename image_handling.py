import cv2
import numpy as np


def save_pixels_as_image(input_layer, image_shape, image_name):
    image_pixels = input_layer.reshape(image_shape)
    image_pixels = image_pixels.astype(np.uint8)
    cv2.imwrite(image_name, image_pixels)


def visualize_pixels(input_layer, image_shape, label):
    input_layer *= 255
    image_pixels = input_layer.reshape(image_shape)
    image_pixels = image_pixels.astype(np.uint8)
    cv2.imshow(label, image_pixels)
    cv2.waitKey(0)

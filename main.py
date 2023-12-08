import neural_network
import numpy as np
import cv2
import os
import image_handling


def main():
	user_choice = input("(1) Train model\n(2) Evaluate model\n(3) Predict value\n\nWhat you wanna do? User: ")
	if user_choice == "1":
		print("you choice: training a model\n")
		neural_network.train_model(
			train_dir="Dataset\\train", hidden_layers=(16, 16),
			output_layer=10, epochs=20, batch_size=100)
	elif user_choice == "2":
		print("you choice: evaluating a model\n")
		neural_network.evaluate()
	elif user_choice == "3":
		print("you choice: predicting an output\n")
		image = cv2.imread("Images\\Test\\4.png")
		input_layer = 0
		if image is not None:
			input_layer = image.flatten()
		model_path = neural_network.ui_choose_model()
		weights, biases = neural_network.import_model(model_path)
		neural_network.predict(input_layer, weights, biases)
	else:
		print("Answer must be 1, 2 or 3")


if __name__ == "__main__":
	main()

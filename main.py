import neural_network
import numpy as np


def main():
	user_choice = input("(1) Train model\n(2) Evaluate model\n(3) Predict value\n\nWhat your wanna do?\nUser: ")
	if user_choice == "1":
		print("you choice: training a model\n")
		neural_network.train_model(
			train_dir="Dataset\\train", hidden_layers=(16, 16),
			output_layer=10, epochs=20, batch_size=100)
	elif user_choice == "2":
		print("you choice: evaluating a model\n")
		neural_network.evaluate(test_dir="Dataset\\test")
	elif user_choice == "3":
		print("you choice: predicting an output\n")
		neural_network.predict(image_path="Dataset\\test")
	else:
		print("Answer must be 1, 2 or 3")


if __name__ == "__main__":
	main()

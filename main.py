import neural_network


def main():
	user_choice = input("(1) Train model\n(2) Evaluate model\n(3) Predict value\n")
	if (user_choice == "1"):
		neural_network.train_model(train_src="", input_size=784, hidden_layers=(16,16), output_layer=10, epochs=10)
	elif (user_choice == "2"):
		neural_network.evaluate(model_path="", test_dir="")
	elif (user_choice == "3"):
		neural_network.outputpredict(image_path, model_path)
	else:
		print("Answer must be 1, 2 or 3")


if __name__ == "__main__":
	main()


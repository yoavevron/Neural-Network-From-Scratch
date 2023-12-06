import neural_network

def main():
	user_choice = input("(1) Train model\n(2) Predict value\n")
	if (user_choice == "1"):
		w, b = neural_network.train_model(train_src="",
										  input_size=784,
										  hidden_layers=(16,16),
										  output_layer=10,
										  epochs=10)
		neural_network.evaluate(w, b, test_dir)
	elif (user_choice == 2):
		neural_network.predict(image_path, model_path)

if __name__ == "__main__":
	main()


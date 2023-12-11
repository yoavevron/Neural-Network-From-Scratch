import image_handling
import initial_model_values
import activation_functions
import loss_functions
import optimizers
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime


def train_model(train_dir, hidden_layers, output_layer, epochs, batch_size, learning_rate):

	# Load the data then shuffle it
	src_images, src_labels, input_size = load_data(train_dir)  # Read train data from csv to arrays
	permutation = np.random.permutation(len(src_labels))
	images = src_images[permutation]
	labels = src_labels[permutation]

	# Initialize random weights and biases
	w = initial_model_values.random_weights_init(input_size, hidden_layers, output_layer)
	b = initial_model_values.random_bias_init(hidden_layers, output_layer)

	num_samples = images.shape[0]

	for epoch in tqdm(range(epochs), leave=True):
		total_loss = 0
		correct_predictions = 0
		cost = 0
		for start in range(0, num_samples, batch_size):
			end = start + batch_size
			batch_images = images[start:end]
			batch_labels = labels[start:end]
			n = initial_model_values.load_neurons(batch_images, hidden_layers, output_layer, batch_size)

			# Forward propagation
			z, n = forward_propagation(n, w, b)

			# Calculate loss
			model_prediction = n[-1]
			cost += loss_functions.categorical_cross_entropy(model_prediction, batch_labels)

			# Calculate accuracy
			batch_predicted = np.argmax(activation_functions.softmax(n[-1]), axis=1)
			batch_labels_as_integers = batch_labels.astype(int)
			correct_predictions += (np.sum(batch_predicted == batch_labels_as_integers))

			# Backward propagation
			w, b = optimizers.gradient_descent(a=n, z=z, w=w, b=b, x=z[0], label=batch_labels_as_integers,
											   m=num_samples/batch_size, learn_rate=learning_rate)
			# print(w[-1][-1])
		average_loss = cost / num_samples
		average_accuracy = correct_predictions * 100 / num_samples
		print(f"Epoch {epoch}/{epochs}. Loss: {average_loss}, Accuracy: {average_accuracy}%")
		total_loss = 0
		correct_predictions = 0
	export_model(w, b)
	return w, b


def load_data(src):
	"""
	Load all the supervised train data from the directory provided into two np arrays
	:param src: directory path of the train data (csv files)
	:return: a numpy array of train data, a numpy array of train labels, input layer size
	"""
	print("Dataset load started...")
	all_files = os.listdir(src)
	csv_files = [file for file in all_files if file.endswith('.csv')]
	images = []
	labels = np.array([])
	for csv_file in csv_files:
		file_path = os.path.join(src, csv_file)
		df = pd.read_csv(file_path)
		label = np.array(df['label'])
		labels = np.concatenate((labels, label))
		image = df.iloc[:, 1:]
		images.append(image)
	images_df = pd.concat(images)
	images = images_df.to_numpy()
	input_size = images.shape[1]
	print("Dataset load finished!")
	print("Input layer size is {0} according to the train data.".format(input_size))
	return images, labels, input_size


def forward_propagation(n, w, b):
	"""
	Run the feed forward process using the weights, biases and activation function
	to calculate the output of the model for the neurons that were inserted
	For now I set for this model relu for all layers except the last one which has no activation function
	:param n: a list of 1d arrays represent the neurons of the model.
	first layer is image pixels, all the others initially zeros
	:param w: weights of the model to compute output
	:param b: biases of the model to compute output
	:return: a softmax probabilities vector of the output to the inserted input image
	"""
	z = []
	z.append(n[0])

	for i in range(1, len(n)-1):
		n[i] = n[i-1].dot(w[i-1])
		n[i] = n[i] + b[i-1]
		z.append(n[i])
		n[i] = activation_functions.relu(n[i])
	# Calculate the last layer separately to avoid activate it with relu
	n[-1] = n[-2].dot(w[-1])
	n[-1] = n[-1] + b[-1]
	z.append(n[-1])
	n[-1] = activation_functions.softmax(n[-1])
	return z, n


def export_model(weights, biases):
	"""
	export the weights and biases for future usages
	:param weights: resulted output of the model
	:param biases: resulted biases of the model
	:return: create a directory for the model and save the weights and biases in different files
	"""
	model_name = str(datetime.now())[:19]
	new_model_name = model_name.replace(":", "_")
	new_model_name = new_model_name.replace(" ", "_")
	new_model_name = new_model_name.replace("-", "_")
	model_directory = os.path.join("models", new_model_name)
	if not os.path.exists(model_directory):
		os.makedirs(model_directory)
	output_weights_file_name = os.path.join(model_directory, "weights.csv")
	output_bias_file_name = os.path.join(model_directory, "biases.csv")
	np.savetxt(output_weights_file_name, weights, delimiter=',')
	np.savetxt(output_bias_file_name, biases, delimiter=',')


def evaluate():
	"""
	Compute the accuracy of the model by predicting all the test data and compare to the real labels
	:return: accuracy rank as a scalar of percents (0-bad, 100-good)
	"""
	model_path = ui_choose_model()
	weights, biases = import_model(model_path)
	test_images, test_labels, input_size = load_data(src="Dataset\\test")
	correct_predictions = 0
	for i in range(len(test_labels)):
		prediction = predict(test_images[i], weights, biases)
		if prediction == test_labels[i]:
			correct_predictions += 1
	model_accuracy = str(correct_predictions * 100 / len(test_labels))[:6]
	print(f"\n{model_path} accuracy is: {model_accuracy}")
	''' IMPLEMENT - export evaluation excel of every correct/incorrect image prediction and total accuracy result'''


def predict(input_layer, weights, biases):
	"""
	Get a model (weight and biases) and a photo and run forward propagation to get a prediction
	:param input_layer: 1d nparray represent the image pixels
	:param weights: 1d nparray represent the model weights
	:param biases: 1d nparray represent the model biases
	:return: the predicted output of the model
	"""
	neurons = initial_model_values.load_neurons(
		img=input_layer, hidden_layers=np.array([len(arr) for arr in biases[:-1]]), output_layer=len(biases[-1]))
	neurons, neurons_before_active = forward_propagation(neurons, weights, biases)
	model_result = np.argmax(activation_functions.softmax(neurons[-1]))
	print(f"The prediction for that image is: {model_result}")
	visualize_result(img=input_layer, predicted_output=model_result)
	return model_result


def ui_choose_model():
	"""
	Ui interface for the user to choose model
	:return: the model name that he chose from the models folder
	"""
	models = os.listdir("models")
	print("\nMY MODELS:")
	for i in range(len(models)):
		print(f"({i}) {models[i]}")
	model_index = str(input("\nPlease choose a model for evaluation: "))
	print(f"Your choice: {models[int(model_index)]}")
	model = models[int(model_index)]
	model_path = os.path.join("models", model)
	return model_path


def import_model(model_path):
	"""
	Load the weights and biases of a model by its path
	:param model_path: the path of the model
	:return: the weights and biases of the model
	"""
	weights_path = os.path.join(model_path, "weights.csv")
	biases_path = os.path.join(model_path, "biases.csv")
	weights = pd.read_csv(weights_path, header=None)
	biases = pd.read_csv(biases_path, header=None)
	weights = weights.values
	biases = biases.values
	return weights, biases


''' IMPLEMENT '''


def visualize_result(img, predicted_output):
	print(img)
	print(predicted_output)
	return "draw an image with its label"


def draw_network(img, neurons, weights, predicted_output, real_output):
	"""
	a function that draw the activation neurons of a network
	:return:
	"""
	print(img, neurons, weights, predicted_output, real_output)
	return 1

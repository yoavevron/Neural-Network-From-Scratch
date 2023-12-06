import initial_model_values
import activation_function
import loss_functions
import optimizers
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import datetime

def train_model(train_dir, hidden_layers, output_layer, epochs):
	"""
	Main training function that run all the necessary calculations
	:param train_src:
	:param input_size:
	:param hidden_layers:
	:param output_layer:
	:param epochs:
	:return:
	"""
	images, labels, input_size = load_data(train_dir)  # Read train data from csv to arrays
	return
	weights = initial_model_values.xavier_weights_init(input_size, hidden_layers, output_layer)
	biases = initial_model_values.xavier_bias_init(input_size, hidden_layers, output_layer)
	for epoch in tqdm(epochs):
		for img in images:
			neurons = initial_model_values.load_neurons(input_size)
			'''Run the network to get predicted value'''
			model_prediction = forward_propagation(neurons, weights, biases)

			'''Calculate a scalar of diff between prediction and real val'''
			cost = loss_functions.cross_entropy(model_prediction, labels[images.indexof(img)])

			'''Run the network backwards to calculate gradients
			Optimize the weights and biases with the gradients'''
			weights, bias = optimizers.adam(cost, neurons, weights, biases, hidden_layers)
		print(cost)
	export_model(weights, bias)
	return weights, bias


def export_model(weights, bias):
	output_file_name = "model_" + str(datetime.now())
	f = open(output_file_name, "w")
	f.write(str(weights))
	f.write(str(bias))
	f.close()


def load_data(train_src):
	"""
	Load all the supervised train data from the directory provided into two np arrays
	:param train_src: directory path of the train data (csv files)
	:return: a numpy array of train data, a numpy array of train labels, input layer size
	"""
	print("Dataset load started...")
	all_files = os.listdir(train_src)
	csv_files = [file for file in all_files if file.endswith('.csv')]
	images = []
	labels = np.array([])
	for csv_file in csv_files:
		file_path = os.path.join(train_src, csv_file)
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


def evaluate(model_path, test_dir):
	print('hi')
# Test the data on predict many times to calculate accuracy
# Export excel of the image, predicted, real value using pandas
# Save all the test images with predicted and real value using matplotlib


def forward_propagation(neurons, weights, biases):
	for i in range(len(weights)):
		for n in neurons[i+1]:
			neurons[i+1][n] = activation_function.relu(weights[0].multiple(neurons)+biases[i][n])
		# all a.function are relu except last->softmax
		return neurons, max(neurons[-1])


def predict(image_path, model_path):
	weights, biases = import_model(model_path)
	neurons = initial_model_values.load_neurons(image_path)
	forward_propagation(neurons, weights, biases)


def import_model(model_path):
	data_frame = pd.read_csv(model_path)
	return data_frame.loc[0], data_frame.loc[1]


def visualize_result(img, neurons, weights, predicted_output, real_output):
	print("draw")
	return "drawing"

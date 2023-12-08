import initial_model_values
import activation_functions
import loss_functions
import optimizers
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import datetime


def train_model(train_dir, hidden_layers, output_layer, epochs, batch_size):
	images, labels, input_size = load_data(train_dir)  # Read train data from csv to arrays
	weights = initial_model_values.random_weights_init(input_size, hidden_layers, output_layer)
	biases = initial_model_values.random_bias_init(hidden_layers, output_layer)

	num_samples = images.shape[0]
	for epoch in tqdm(range(epochs)):
		indices = np.arange(num_samples)
		np.random.shuffle(indices)
		batch_loss = 0

		for start in tqdm(range(0, num_samples, batch_size)):
			end = start + batch_size
			batch_indices = indices[start:end]

			images_batch = images[batch_indices]
			labels_batch = labels[batch_indices]

			for i in range(batch_size):
				neurons = initial_model_values.load_neurons(images_batch[i], hidden_layers, output_layer)

				'''Run the network to get predicted value'''
				neurons, neurons_before_active = forward_propagation(neurons, weights, biases)
				model_prediction = neurons[-1]

				'''Calculate a scalar of diff between prediction and real val'''
				cost = loss_functions.categorical_cross_entropy(model_prediction, labels_batch[i])
				batch_loss += cost

				old = weights.copy()
				'''Run the network backwards to calculate gradients
				Optimize the weights and biases with the gradients'''
				weights, biases = optimizers.stochastic_gradient_descent(
					img_pixels=neurons[0], neurons_before=neurons_before_active, neurons_after=neurons[1:],
					weights=weights, biases=biases, real=labels_batch[i], learning_rate=0.001)
				# weights, biases = optimizers.backward_pass(
				# 	X=images_batch[i], y_true=labels_batch[i], hidden_inputs=neurons_before_active[1:],
				# 	hidden_outputs=neurons[1:], weights=weights, biases=biases, learning_rate=0.001)
		print("Epoch: {0}.\t loss: {1}". format(epoch, batch_loss/batch_size))
	return
	export_model(weights, biases)
	return weights, biases


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


def forward_propagation(neurons, weights, bias):
	"""
	Run the feed forward process using the weights, biases and activation function
	to calculate the output of the model for the neurons that were inserted
	For now I set for this model relu for all layers except the last one which has not activation function
	:param neurons: a list of 1d arrays represent the neurons of the model. first layer is image pixels, all the others initially zeros
	:param weights: weights of the model to compute output
	:param bias: biases of the model to compute output
	:return: a softmax probabilities vector of the output to the inserted input image
	"""
	neurons_before_active = []
	for i in range(1, len(neurons)-1):
		neurons[i] = neurons[i-1].dot(weights[i-1])
		neurons[i] = neurons[i] + bias[i-1]
		neurons_before_active.append(neurons[i])
		neurons[i] = activation_functions.relu(neurons[i])
	neurons[-1] = neurons[-2].dot(weights[-1])
	neurons[-1] = neurons[-1] + bias[-1]
	neurons_before_active.append(neurons[-1])
	return neurons, neurons_before_active


def export_model(weights, bias):
	output_file_name = "model_" + str(datetime.now())
	f = open(output_file_name, "w")
	f.write(str(weights))
	f.write(str(bias))
	f.close()


def evaluate(model_path, test_dir):
	print('hi')
# Test the data on predict many times to calculate accuracy
# Export excel of the image, predicted, real value using pandas
# Save all the test images with predicted and real value using matplotlib


def predict(image_path, model_path):
	# print(np.argmax(model_prediction_vector)) after getting output
	weights, biases = import_model(model_path)
	neurons = initial_model_values.load_neurons(image_path)
	forward_propagation(neurons, weights, biases)


def import_model(model_path):
	data_frame = pd.read_csv(model_path)
	return data_frame.loc[0], data_frame.loc[1]


def visualize_result(img, neurons, weights, predicted_output, real_output):
	print("draw")
	return "drawing"

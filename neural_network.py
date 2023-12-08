import initial_model_values
import activation_functions
import loss_functions
import optimizers
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import cv2


def train_model(train_dir, hidden_layers, output_layer, epochs, batch_size):
	images, labels, input_size = load_data(train_dir)  # Read train data from csv to arrays
	weights = initial_model_values.random_weights_init(input_size, hidden_layers, output_layer)
	biases = initial_model_values.random_bias_init(hidden_layers, output_layer)

	num_samples = images.shape[0]
	for epoch in tqdm(range(epochs)):
		total_loss = 0
		indices = np.arange(num_samples)
		np.random.shuffle(indices)

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
				total_loss += cost

				'''Run the network backwards to calculate gradients
				Optimize the weights and biases with the gradients'''
				weights, biases = optimizers.stochastic_gradient_descent(
					img_pixels=neurons[0], neurons_before=neurons_before_active, neurons_after=neurons[1:],
					weights=weights, biases=biases, real=labels_batch[i], learning_rate=0.001)
		print("Epoch: " + str(epoch) + ", loss: " + str(total_loss/batch_size))
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


def export_model(weights, biases):
	"""
	export the weights and biases for future usages
	:param weights: resulted output of the model
	:param biases: reuslted biases of the model
	:return: create a directory for the model and save the weights and biases in differnet files
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


def evaluate(test_dir):
	'''Test the data on predict many times to calculate accuracy
	Export excel of the image, predicted, real value using pandas
	Save all the test images with predicted and real value using matplotlib'''
	model = ui_choose_model()
	# Import the model
	# load the test data
	# predict all the images
	# sum the process' accuarcy


def predict(image_path):
	"""
	Get a model (weight and biases) and a photo and run forward propagation to get a prediction
	:param image_path: path to the image for prediction
	:param model_path: path to the model that you want the user want to use for prediction
	:return: the predicted output of the model
	"""
	model = ui_choose_model()
	model_path = os.path.join("models", model)
	weights, biases = import_model(model_path)
	image = cv2.imread(image_path)
	input_layer = 0
	if image is not None:
		input_layer = image.flatten()
	neurons = initial_model_values.load_neurons(
		img=input_layer, hidden_layers=np.array([len(arr) for arr in biases[:-1]]),output_layer=len(biases[-1]))
	neurons, neurons_before_active = forward_propagation(neurons, weights, biases)
	model_result = np.argmax(activation_functions.softmax(neurons[-1]))
	print(f"The prediction for that image is: {model_result}")
	visualize_result(img=image, predicted_output=model_result)
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
	return models[int(model_index)]


def import_model(model_path):
	data_frame = pd.read_csv(model_path)
	return data_frame.loc[0], data_frame.loc[1]


''' IMPLEMENT '''


def visualize_result(img, predicted_output):
	return "draw an image with its label"


def draw_network(img, neurons, weights, predicted_output, real_output):
	"""
	a function that draw the activation neurons of a network
	:return:
	"""
	return 1

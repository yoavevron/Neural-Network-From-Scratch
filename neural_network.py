import initial_model_values
import activation_function
import loss_functions
import optimizers
import pandas as pd
from tqdm import tqdm
import datetime

def train_model(train_src, input_size, hidden_layers, output_layer, epochs):
	"""
	Main training function that run all the necessary calculations
	:param train_src:
	:param input_size:
	:param hidden_layers:
	:param output_layer:
	:param epochs:
	:return:
	"""
	images, labels = load_data(train_src)  # Read train data from csv to arrays
	weights = initial_model_values.xavier_weights_init(input_size, hidden_layers, output_layer)
	biases = initial_model_values.xavier_bias_init(input_size, hidden_layers, output_layer)
	for epoch in tqdm(epochs):
		for img in images:
			neurons = initial_model_values.load_neurons(img)
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
	print("implement")
	return "images", "labels"


def evaluate(w, b, test_dir):
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


def predict(img, weights, biases):
	neurons = initial_model_values.load_neurons(img)
	forward_propagation(neurons, weights, biases)


def import_model(model_path):
	data_frame = pd.read_csv(model_path)
	return data_frame.loc[0], data_frame.loc[1]


def visualize_result(img, neurons, weights, predicted_output, real_output):
	print("draw")
	return "drawing"

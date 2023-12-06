import initial_model_values

def train_model(train_src, input_size, hidden_layers, output_layer, epochs):
    '''
    Main training function that run all the necessary calculations
    :param train_src:
    :param input_size:
    :param hidden_layers:
    :param output_layer:
    :param epochs:
    :return:
    '''
	images, labels = load_data(train_src) # Read train data from csv to arrays
    weights = xavier/ha_init_weights(layers)
    biases = xavier/ha_init_biases(layers)
    For epoch in tqdm(epochs):
		For img in images:
			# Run the network to get predicted value
			Predicted = feed_forward(img, weights, biases)
			# Calculate a scalar of diff between prediction and real val
			cost = loss_function(predicted, labels[images.indexof(img)])
			# Run the network backwards to calculate gradients
			gradients = compute_gradients(cost, neurons, weights, biases)
            # Optimize the weights and biases with the gradients
			Weights, biases = optimize(img, weights, biases, layers)
		Print(loss)
    Export_model(weights, biases)
    Return weights, biases

def export_model(weights, biases)
	Write_file(weights, biases)

def load_data(train_src):
    print("implement")
    return "images", "labels"

Def evaluate(w, b, test_dir):
# Test the data on predict many times to calculate accuracy
# Export excel of the image, predicted, real value using pandas
# Save all the test images with predicted and real value using matplotlib

Def Forward_propogation(img, weights, biases)
neurons = generate_neruns_values(img)
For i in range(len(weights):
	For n in neurons[i+1]:
		neurons[i+1][n] =
activation_function(weights[0].multiple(neurons)+biases[i][n])
# all a.function are relu except last->softmax
	return neurons, max(neurons[-1])

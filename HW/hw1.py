import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights1)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights2)
        self.output = self.sigmoid(self.output_layer_input)
        return self.output
    
    def backward(self, inputs, target, learning_rate):
        error = target - self.output
        d_output_layer = error * self.sigmoid_derivative(self.output)
        error_hidden_layer = np.dot(d_output_layer, self.weights2.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer_output)
        self.weights2 += learning_rate * np.dot(self.hidden_layer_output.T, d_output_layer)
        self.weights1 += learning_rate * np.dot(inputs.T, d_hidden_layer)

if __name__ == "__main__":
    m = int(input("Enter the number of neurons in the input layer: "))
    n = int(input("Enter the number of neurons in the hidden layer: "))
    inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    target = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
    nn = NeuralNetwork(input_size=3, hidden_size=n,output_size=2)
    for i in range(1500):
        output = nn.forward(inputs)
        nn.backward(inputs, target, learning_rate=0.1)
    print("Final output after training: ", output)
    print(nn.forward([0,1,0]))

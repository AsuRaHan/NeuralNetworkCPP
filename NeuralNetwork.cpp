#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) {
    for (int i = 1; i < layer_sizes.size(); i++) {
        layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
    }
}

std::vector<double> NeuralNetwork::feedforward(const std::vector<double>& inputs) {
    std::vector<double> outputs = inputs;
    for (int i = 0; i < layers.size(); i++) {
        outputs = layers[i].feedforward(outputs);
    }
    return outputs;
}
void NeuralNetwork::backpropagate(std::vector<double> input, std::vector<double> target) {
    // 1. Вычисление выходного значения каждого нейрона в сети для заданного входного вектора.
    std::vector<double> hidden_outputs(hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        double activation = 0.0;
        for (int j = 0; j < input_size; j++) {
            activation += input[j] * ih_weights[j][i];
        }
        activation += ih_biases[i];
        hidden_outputs[i] = sigmoid(activation);
    }

    std::vector<double> outputs(output_size);
    for (int i = 0; i < output_size; i++) {
        double activation = 0.0;
        for (int j = 0; j < hidden_size; j++) {
            activation += hidden_outputs[j] * ho_weights[j][i];
        }
        activation += ho_biases[i];
        outputs[i] = sigmoid(activation);
    }

    // 2. Вычисление ошибки выходного слоя сети на основе ожидаемых выходных значений и фактических выходных значений.
    std::vector<double> output_errors(output_size);
    for (int i = 0; i < output_size; i++) {
        output_errors[i] = (target[i] - outputs[i]) * sigmoid_derivative(outputs[i]);
    }

    // 3. Распространение ошибки на предыдущие слои сети, вычисление ошибки для каждого нейрона и корректировка весов связей.
    std::vector<double> hidden_errors(hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        double error = 0.0;
        for (int j = 0; j < output_size; j++) {
            error += output_errors[j] * ho_weights[i][j];
        }
        hidden_errors[i] = error * sigmoid_derivative(hidden_outputs[i]);
    }

    // корректировка весов связей между скрытым и выходным слоями
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < output_size; j++) {
            ho_weights[i][j] += learning_rate * output_errors[j] * hidden_outputs[i];
        }
        ho_biases[i] += learning_rate * output_errors[i];
    }

    // корректировка весов связей между входным и скрытым слоями
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            ih_weights[i][j] += learning_rate * hidden_errors[j] * input[i];
        }
        ih_biases[i] += learning_rate * hidden_errors[i];
    }
}

void NeuralNetwork::backpropagate(const std::vector<double>& error, double learning_rate) {
    // Calculate output layer errors
    for (int i = 0; i < output_layer_.size(); ++i) {
        double output = output_layer_[i]->get_output();
        double error_gradient = output * (1.0 - output) * error[i];
        output_layer_[i]->set_error_gradient(error_gradient);
    }

    // Calculate hidden layer errors
    for (int i = 0; i < hidden_layer_.size(); ++i) {
        double output = hidden_layer_[i]->get_output();
        double error_gradient = output * (1.0 - output) * output_layer_[0]->get_weight(i) * output_layer_[0]->get_error_gradient();
        hidden_layer_[i]->set_error_gradient(error_gradient);
    }

    // Update output layer weights
    for (int i = 0; i < output_layer_.size(); ++i) {
        double output = output_layer_[i]->get_output();
        double delta = learning_rate * output_layer_[i]->get_error_gradient();
        output_layer_[i]->update_weights(delta);
    }

    // Update hidden layer weights
    for (int i = 0; i < hidden_layer_.size(); ++i) {
        double output = hidden_layer_[i]->get_output();
        double delta = learning_rate * hidden_layer_[i]->get_error_gradient();
        hidden_layer_[i]->update_weights(delta);
    }
}

// функция активации sigmoid
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// производная функции активации sigmoid
double NeuralNetwork::sigmoid_derivative(double x) {
    return x * (1.0 - x);
}
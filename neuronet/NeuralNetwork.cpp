#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size) {
    this->input_size = input_size;
    this->hidden_layer = Layer(input_size, hidden_size);
    this->output_layer = Layer(hidden_size, output_size);
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& inputs) {
    auto hidden_outputs = hidden_layer.forward(inputs);
    return output_layer.forward(hidden_outputs);
}

void NeuralNetwork::train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets, double learning_rate, int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int j = 0; j < inputs.size(); j++) {
            // Прямое распространение
            std::vector<double> hidden_outputs = hidden_layer.forward(inputs[j]);
            std::vector<double> outputs = output_layer.forward(hidden_outputs);
            // Вычисление ошибки
            std::vector<double> output_errors(output_size);
            for (int k = 0; k < output_size; k++) {
                output_errors[k] = targets[j][k] - outputs[k];
            }
            std::vector<double> hidden_errors(hidden_layer.get_biases().size());
            for (int k = 0; k < hidden_layer.get_biases().size(); k++) {
                double error = 0.0;
                for (int l = 0; l < output_size; l++) {
                    error += output_errors[l] * output_layer.get_weights()[k * output_size + l];
                }
                hidden_errors[k] = error * hidden_outputs[k] * (1.0 - hidden_outputs[k]);
            }
            // Обновление весов и смещений
            std::vector<double> output_weights = output_layer.get_weights();
            std::vector<double> output_biases = output_layer.get_biases();
            for (int k = 0; k < output_size; k++) {
                for (int l = 0; l < hidden_outputs.size(); l++) {
                    output_weights[k * hidden_outputs.size() + l] += learning_rate * output_errors[k] * hidden_outputs[l];
                }
                output_biases[k] += learning_rate * output_errors[k];
            }
            output_layer.set_weights(output_weights);
            output_layer.set_biases(output_biases);
            std::vector<double> hidden_weights = hidden_layer.get_weights();
            std::vector<double> hidden_biases = hidden_layer.get_biases();
            for (int k = 0; k < hidden_layer.get_biases().size(); k++) {
                for (int l = 0; l < input_size; l++) {
                    hidden_weights[k * input_size + l] += learning_rate * hidden_errors[k] * inputs[j][l];
                }
                hidden_biases[k] += learning_rate * hidden_errors[k];
            }
            hidden_layer.set_weights(hidden_weights);
            hidden_layer.set_biases(hidden_biases);
        }
    }
}

std::vector<double> NeuralNetwork::predict(std::vector<double> inputs) {
    std::vector<double> hidden_outputs = hidden_layer.forward(inputs);
    std::vector<double> outputs = output_layer.forward(hidden_outputs);
    return outputs;
}
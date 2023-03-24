#pragma once

#include "Layer.h"

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    std::vector<double> forward(const std::vector<double>& inputs);
    void backward(const std::vector<double>& inputs, const std::vector<double>& outputs);
    void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets, double learning_rate, int num_epochs);
    std::vector<double> predict(std::vector<double> inputs);
private:
    int input_size;
    Layer hidden_layer;
    Layer output_layer;
    int output_size = output_layer.get_biases().size();
};

